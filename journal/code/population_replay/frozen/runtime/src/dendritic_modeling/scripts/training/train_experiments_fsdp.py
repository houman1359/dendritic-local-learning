#!/usr/bin/env python
"""
Large Model FSDP Training Script
================================

This script enables FSDP (Fully Sharded Data Parallel) training for large dendritic models
that don't fit on a single GPU.

Usage:
    torchrun --nproc_per_node=<num_gpus> train_experiments_fsdp.py <config_path>

Example:
    torchrun --nproc_per_node=4 train_experiments_fsdp.py configs/large_model_config.yaml
"""

import argparse
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timedelta

wandb = None  # wandb is not used in this project (removed 2026-08-20)

os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_TIMEOUT", "1800")  # 30 minutes timeout
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault(
    "NCCL_SOCKET_FAMILY", "AF_INET"
)  # Force IPv4 to fix socket errors
os.environ.setdefault("NCCL_SOCKET_IFNAME", "ib0,eth0")
os.environ.setdefault("NCCL_IB_DISABLE", "0")  # Enable InfiniBand if available
os.environ.setdefault("NCCL_NET_GDR_LEVEL", "2")  # Enable GPU Direct RDMA

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.config import load_config
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.datasets.dimensions import flat_dataset_input_dim
from dendritic_modeling.models import Classifier, Regressor
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    collect_model_data_driven_reactivation_policies,
    is_data_driven_reactivation_policy,
    normalize_reactivation_init_policy,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.script_utils import config_utils as _config_utils
from dendritic_modeling.scripts.script_utils.setup_utils import (
    _apply_initialization_seed_defaults,
    _apply_topology_seed_defaults,
    _config_to_plain_dict,
    _get_dataset_specific_params,
    _validate_sealed_test_dataset,
    initialize_model,
)
from dendritic_modeling.scripts.training.train_encoder_network import (
    load_train_encoder_network,
)
from dendritic_modeling.training.dataloader_utils import (
    DataLoaderTuning,
    resolve_dataloader_tuning_from_config,
    seeded_dataloader_kwargs,
    visible_cpu_count,
)
from dendritic_modeling.training.fsdp_utils import get_fsdp_config, save_fsdp_checkpoint
from dendritic_modeling.training.strategies.fsdp_standard import (
    FSDPTrainer,
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.utils import resolve_experiment_seeds
from dendritic_modeling.utils.general import save_dict, set_seed
from dendritic_modeling.utils.logging_config import LoggerManager

logger = logging.getLogger(__name__)
logger_manager = LoggerManager()
_PROCESS_GROUP_TIMEOUT_ENV = "DENDRITIC_FSDP_PROCESS_GROUP_TIMEOUT_SECONDS"
canonicalize_model_core_flags = getattr(
    _config_utils, "canonicalize_model_core_flags", None
)
canonicalize_rnn_core_flags = getattr(
    _config_utils, "canonicalize_rnn_core_flags", None
)

WANDB_AVAILABLE = False

# torchinfo import (optional)
try:
    import torchinfo

    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    logger.warning("torchinfo not available. Model summary will be skipped.")


def setup_wandb(config, rank, world_size):
    """Initialize WandB logging (only on rank 0)."""
    if rank != 0:
        return None

    wandb_config = config.wandb
    if not wandb_config.use_wandb:
        return None

    raise RuntimeError("wandb is disabled in this project")


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed environment via ``torchrun``.

    ``torchrun`` sets ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE``
    automatically.  When these are absent the function raises an error
    because FSDP requires a distributed process group.

    Returns:
        (rank, local_rank, world_size)
    """
    if "RANK" not in os.environ:
        raise RuntimeError(
            "FSDP requires torchrun.  Launch with:\n"
            "  torchrun --nproc_per_node=<N> train_experiments_fsdp.py <config>"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not dist.is_initialized():
        timeout_seconds_raw = os.environ.get(_PROCESS_GROUP_TIMEOUT_ENV, "1800")
        try:
            timeout_seconds = int(timeout_seconds_raw)
        except ValueError as exc:
            raise ValueError(
                f"{_PROCESS_GROUP_TIMEOUT_ENV} must be a positive integer, "
                f"got {timeout_seconds_raw!r}"
            ) from exc
        if timeout_seconds <= 0:
            raise ValueError(
                f"{_PROCESS_GROUP_TIMEOUT_ENV} must be a positive integer, "
                f"got {timeout_seconds_raw!r}"
            )
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=timeout_seconds),
        )

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _save_fsdp_run_config(config, run_save_path: str) -> None:
    """Persist the resolved FSDP run config with legacy flag canonicalization."""
    config_payload = asdict(config)
    if canonicalize_model_core_flags is not None:
        config_payload = canonicalize_model_core_flags(config_payload)
    if canonicalize_rnn_core_flags is not None:
        config_payload = canonicalize_rnn_core_flags(config_payload)
    save_dict(config_payload, run_save_path, "config.json")


def _setup_rank0_run_directory(config) -> str:
    """Create the rank-0 output directory, logger file, and config snapshot."""
    if bool(getattr(config.outputs, "exact_run_dir", False)):
        run_save_path = config.outputs.results_dir
    else:
        run_id = f"{config.outputs.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_save_path = os.path.join(config.outputs.results_dir, run_id)
    os.makedirs(run_save_path, exist_ok=True)

    logger_manager.set_log_directory(run_save_path)
    log_file = os.path.join(run_save_path, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    _save_fsdp_run_config(config, run_save_path)
    return run_save_path


def print_model_parameter_summary(model, sample_input=None):
    """Print detailed model parameter summary using torchinfo if available."""
    logger.info("Network architecture summary before training:")
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        is_sequence_input = (
            isinstance(sample_input, torch.Tensor) and sample_input.dim() >= 2
        )
        summary_is_safe = total_params <= 10_000_000 and not is_sequence_input

        if TORCHINFO_AVAILABLE and sample_input is not None and summary_is_safe:
            torchinfo.summary(
                model,
                input_size=(1, *sample_input.shape),
                col_names=["input_size", "output_size", "num_params"],
                col_width=25,
                row_settings=["depth"],
                depth=7,
            )
            logger.info("Model summary generated (see console output)")
        else:
            if not TORCHINFO_AVAILABLE:
                logger.warning(
                    "torchinfo is not available. Model summary not generated."
                )
            elif not summary_is_safe:
                logger.info(
                    "Skipping torchinfo forward for a recurrent or large model; "
                    "reporting parameter counts without an artificial pre-FSDP batch."
                )
            else:
                logger.warning("No sample input provided. Using basic parameter count.")

            # Basic parameter breakdown
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

    except Exception as e:
        logger.error(f"Failed to generate model summary before training: {e}")
        # Fallback to basic count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")


def _print_rank0_model_summary(model, train_ds) -> None:
    """Generate the rank-0 model summary, falling back when sampling fails."""
    try:
        sample_input = train_ds[0][0]
        print_model_parameter_summary(model, sample_input)
    except Exception as e:
        logger.warning(f"Could not generate model summary with sample input: {e}")
        print_model_parameter_summary(model)


def _build_fsdp_runtime_configs(config):
    """Build trainer-common, FSDP, and optimizer configs from typed config."""
    train_common = config.training.main.common

    fsdp_cfg = config.distributed.fsdp
    logger.info("Using FSDP configuration from config file")
    fsdp_config = get_fsdp_config(
        mixed_precision=fsdp_cfg.mixed_precision,
        cpu_offload=fsdp_cfg.cpu_offload,
        sharding_strategy=fsdp_cfg.sharding_strategy,
        fsdp_config_dict=vars(fsdp_cfg) if hasattr(fsdp_cfg, "__dict__") else None,
    )

    param_groups = train_common.param_groups
    optimizer_config = {
        "lr": param_groups.lr,
        "weight_decay": getattr(train_common, "weight_decay_rate", 0.0),
        "split_params": getattr(param_groups, "split_params", False),
    }
    # Keep the component-specific learning rates from the typed YAML config.
    # These values are especially important for very wide decoders, where a
    # single global Adam learning rate can produce a width-dependent jump.
    for key in (
        "topk_lr",
        "blocklinear_lr",
        "reactivation_lr",
        "decoder_lr",
        "decoder_input_lr",
        "encoder_lr",
    ):
        value = getattr(param_groups, key, None)
        if value is not None:
            optimizer_config[key] = value
    return train_common, fsdp_config, optimizer_config


def _create_fsdp_trainer(
    *,
    rank: int,
    world_size: int,
    fsdp_config,
    run_save_path,
    train_common,
    optimizer_config,
    analysis_manager,
    wandb_run,
    config,
):
    """Create the FSDP trainer from resolved runtime config."""
    checkpointing = getattr(config.experiment, "checkpointing", {}) or {}
    checkpointing_enabled = bool(checkpointing.get("enabled", False))
    save_every_n_epochs = (
        int(checkpointing.get("save_every_n_epochs", 0) or 0)
        if checkpointing_enabled
        else None
    )
    return FSDPTrainer(
        rank=rank,
        world_size=world_size,
        fsdp_config=fsdp_config,
        checkpoint_dir=run_save_path if rank == 0 else None,
        save_every_n_epochs=save_every_n_epochs,
        optimizer_config=optimizer_config,
        analysis_manager=analysis_manager,
        wandb_run=wandb_run,
        epochs=train_common.epochs,
        loss_function=getattr(train_common, "loss_function", "cat_nll"),
        grad_clip_value=getattr(train_common, "grad_clip_value", 5.0),
        use_amp=getattr(train_common, "use_amp", False),
        early_stopping=getattr(train_common, "early_stopping", False),
        patience=getattr(train_common, "patience", 10),
        print_every=getattr(train_common, "print_every", 1),
        log_first_step_diagnostics=getattr(
            train_common,
            "log_first_step_diagnostics",
            False,
        ),
        cache_frozen_core_outputs=getattr(
            config.distributed.fsdp,
            "cache_frozen_core_outputs",
            False,
        ),
        cached_core_batch_size=getattr(
            config.distributed.fsdp,
            "cached_core_batch_size",
            None,
        ),
        checkpointing=checkpointing_enabled,
        plot_losses=getattr(train_common, "plot_losses", False),
        save_path=run_save_path if rank == 0 else None,
        seed=int(config.experiment.seed),
    )


def _resolve_large_model_input_dim(data_config, model_config=None) -> int:
    """Resolve input size for manually assembled FSDP models.

    Sequence datasets have configuration-dependent feature widths and are not
    covered by the fixed vision-dataset table.  Match the standard model
    builder by honoring an explicit encoder ``params.input_dim`` first.
    """
    encoder_config = getattr(model_config, "encoder", None)
    encoder_params = getattr(encoder_config, "params", None)
    if encoder_params:
        explicit_input_dim = getattr(encoder_params, "input_dim", None)
        if explicit_input_dim is None:
            try:
                explicit_input_dim = dict(encoder_params).get("input_dim")
            except (TypeError, ValueError):
                explicit_input_dim = None
        if explicit_input_dim is not None:
            explicit_input_dim = int(explicit_input_dim)
            if explicit_input_dim <= 0:
                raise ValueError(
                    "model.encoder.params.input_dim must be positive, got "
                    f"{explicit_input_dim}"
                )
            return explicit_input_dim

    return flat_dataset_input_dim(
        getattr(data_config, "dataset_name", None),
        default_input_dim=784,
    )


def create_large_model(config):
    """Create a large dendritic model from configuration."""
    model_config = config.model

    vision_replacement = getattr(model_config, "vision_replacement", None)
    pretrained_replacement = getattr(model_config, "pretrained_replacement", None)
    replacement_enabled = (
        vision_replacement is not None and getattr(vision_replacement, "enabled", False)
    ) or (
        pretrained_replacement is not None
        and getattr(pretrained_replacement, "enabled", False)
    )
    if replacement_enabled:
        model, _ = initialize_model(model_config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model created from vision/pretrained replacement with %s total parameters",
            f"{total_params:,}",
        )
        logger.info("Trainable parameters: %s", f"{trainable_params:,}")
        return model

    input_dim = _resolve_large_model_input_dim(config.data, model_config)

    # Create encoder
    encoder_config = model_config.encoder
    if encoder_config.type.lower() == "identity":
        encoder = Identity(input_dim)
    else:
        encoder_params = (
            _config_to_plain_dict(encoder_config.params)
            if encoder_config.params
            else {}
        )
        encoder_params["input_dim"] = input_dim
        encoder_network = get_architecture(encoder_config.type, encoder_params)
        encoder = (
            encoder_network.encoder
            if hasattr(encoder_network, "encoder")
            else encoder_network
        )

    # Create core network
    core_config = model_config.core
    if hasattr(encoder, "output_dim"):
        core_network = get_architecture(
            core_config.type, core_config, input_dim=encoder.output_dim
        )
    else:
        core_network = get_architecture(
            core_config.type, core_config, input_dim=input_dim
        )

    # Create decoder
    decoder_config = model_config.decoder
    if decoder_config.type.lower() == "identity":
        decoder = Identity(
            core_network.output_dim
            if hasattr(core_network, "output_dim")
            else core_network.excitatory_layer_sizes[-1]
        )
    else:
        decoder_params = (
            _config_to_plain_dict(decoder_config.params)
            if decoder_config.params
            else {}
        )
        decoder_params = {
            key: value for key, value in decoder_params.items() if value is not None
        }
        if hasattr(core_network, "output_dim"):
            decoder_params["input_dim"] = core_network.output_dim
        elif hasattr(core_network, "excitatory_layer_sizes"):
            decoder_params["input_dim"] = core_network.excitatory_layer_sizes[-1]
        decoder = get_architecture(decoder_config.type, decoder_params)

    # Create model based on task
    if model_config.task == "classification":
        model = Classifier(
            encoder,
            core_network,
            decoder,
            learned_output_scale=getattr(model_config, "learned_output_scale", True),
            fixed_output_scale=getattr(model_config, "fixed_output_scale", 1.0),
            output_scale_mode=getattr(model_config, "output_scale_mode", None),
        )
    elif model_config.task == "regression":
        model = Regressor(encoder, core_network, decoder)
    else:
        raise ValueError(f"Unknown task: {model_config.task}")

    # Log model info with detailed breakdown
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model


def _apply_fsdp_trainability_policy(model, fsdp_config) -> int:
    """Apply opt-in FSDP component freezing before wrapping the model."""
    freeze_core = bool(getattr(fsdp_config, "freeze_core", False))
    cache_core = bool(getattr(fsdp_config, "cache_frozen_core_outputs", False))
    if cache_core and not freeze_core:
        raise ValueError(
            "distributed.fsdp.cache_frozen_core_outputs requires freeze_core=true"
        )
    if not freeze_core:
        return 0
    core = getattr(model, "core_network", None)
    if core is None:
        raise ValueError("distributed.fsdp.freeze_core requires model.core_network")
    encoder = getattr(model, "encoder_network", None)
    if (
        cache_core
        and encoder is not None
        and any(parameter.requires_grad for parameter in encoder.parameters())
    ):
        raise ValueError(
            "cache_frozen_core_outputs requires a frozen or parameter-free encoder"
        )
    model._freeze_core_forward = True
    for module in core.modules():
        module._fsdp_frozen_core = True
    frozen_parameters = 0
    for parameter in core.parameters():
        if parameter.requires_grad:
            frozen_parameters += parameter.numel()
            parameter.requires_grad_(False)
    logger.info(
        "FSDP readout-only mode: froze %s core parameters before wrapping",
        f"{frozen_parameters:,}",
    )
    return frozen_parameters


def _get_analysis_encoder_network(encoder_network):
    """Return the encoder module that analysis paths should see."""
    if isinstance(encoder_network, Identity):
        return encoder_network
    if hasattr(encoder_network, "encoder"):
        return Identity(encoder_network.encoder.output_dim)
    return encoder_network


def _run_final_analysis_on_plain_model(
    *,
    config,
    analysis_manager: AnalysisManager,
    full_ckpt_path: str,
    encoder_network,
    local_rank: int,
) -> None:
    """Reload the gathered FSDP checkpoint into a plain model before analysis."""
    try:
        logger.info("Running final analysis on non-FSDP model...")
        plain_model = create_large_model(config)
        ckpt = torch.load(full_ckpt_path, map_location=f"cuda:{local_rank}")
        plain_model.load_state_dict(ckpt["model_state_dict"])
        plain_model = plain_model.to(f"cuda:{local_rank}")
        plain_model.eval()

        plain_model.encoder_network = _get_analysis_encoder_network(encoder_network)

        analysis_manager.model = plain_model
        analysis_manager.run_analysis(filename="final", training=False)
        logger.info("Final analysis completed - figures saved to performance folder")
    except Exception as e:
        logger.error(f"Final analysis failed: {e}")
        import traceback

        logger.error(traceback.format_exc())


def _requested_data_driven_reactivation_policies(config, model) -> set[str]:
    """Return data-driven reactivation policies requested by config or layers."""
    requested_policies = set()
    try:
        reactivation_cfg = config.model.core.reactivation
        init_policy = normalize_reactivation_init_policy(
            getattr(reactivation_cfg, "init_policy", "analytical")
        )
        if is_data_driven_reactivation_policy(init_policy):
            requested_policies.add(init_policy)
    except AttributeError:
        logger.debug(
            "Reactivation config path missing (config.model.core.reactivation); "
            "skipping config-level data-driven init policy detection."
        )

    layer_requested_policies, _saw_layer_policy = (
        collect_model_data_driven_reactivation_policies(model)
    )
    requested_policies.update(layer_requested_policies)

    return requested_policies


def _maybe_calibrate_fsdp_data_driven_reactivation(
    *,
    config,
    model,
    train_dataset,
) -> None:
    """Run FSDP rank-0 data-driven reactivation calibration when requested."""
    requested_policies = _requested_data_driven_reactivation_policies(config, model)
    if not requested_policies:
        return

    logger.warning(
        "FSDP + init_policy=%r: running calibration on rank 0 only. "
        "Parameter writes may not propagate perfectly across ranks "
        "depending on sharding strategy. Verify with the "
        "ReactivationDynamicsAnalyzer if in doubt.",
        ", ".join(sorted(requested_policies)),
    )
    from dendritic_modeling.scripts.training.train_experiments import (
        _maybe_calibrate_data_driven_reactivation,
    )

    _maybe_calibrate_data_driven_reactivation(
        config=config,
        model=model,
        train_dataset=train_dataset,
        is_main=True,
    )


def _setup_rank0_analysis_manager(
    *,
    config,
    model,
    encoder_network,
    encoded_train_ds,
    encoded_valid_ds,
    encoded_test_ds,
    run_save_path,
    local_rank: int,
):
    """Create the rank-0 analysis manager and run pre-training calibration."""
    logger.info("Setting up AnalysisManager...")
    from dendritic_modeling.scripts.training.train_experiments import (
        _validate_sealed_test_analysis,
    )

    experiment_config = getattr(config, "experiment", None)
    if experiment_config is not None:
        _validate_sealed_test_analysis(experiment_config, config.analysis)

    model.encoder_network = _get_analysis_encoder_network(encoder_network)
    logger.info("Set model encoder to %s", type(model.encoder_network).__name__)

    encoded_data = {
        "train": encoded_train_ds,
        "valid": encoded_valid_ds,
        "test": encoded_test_ds,
    }
    manager_kwargs = {
        "model": model,
        "data": encoded_data,
        "analysis_config": config.analysis,
        "save_root": run_save_path,
        "device": f"cuda:{local_rank}",
    }
    if experiment_config is not None:
        analysis_seeds = resolve_experiment_seeds(experiment_config)
        manager_kwargs.update(
            evaluation_seed=analysis_seeds.evaluation_seed,
            probe_seed=analysis_seeds.probe_seed,
        )
    analysis_manager = AnalysisManager(**manager_kwargs)
    logger.info(f"AnalysisManager created: {type(analysis_manager)}")
    logger.info(
        f"Performance analysis enabled: {config.analysis.performance_analysis.enabled}"
    )
    logger.info(
        f"Performance analysis training: {config.analysis.performance_analysis.training}"
    )

    _maybe_calibrate_fsdp_data_driven_reactivation(
        config=config,
        model=model,
        train_dataset=encoded_train_ds,
    )
    return analysis_manager


def _save_final_fsdp_checkpoint_and_run_analysis(
    *,
    rank: int,
    trainer,
    run_save_path,
    config,
    analysis_manager,
    encoder_network,
    local_rank: int,
) -> str:
    """Reuse or save one full checkpoint, then run requested final analysis."""
    analysis_config = getattr(config, "analysis", None)
    analysis_enabled = bool(
        analysis_config is not None
        and hasattr(analysis_config, "any_enabled")
        and analysis_config.any_enabled()
    )
    full_ckpt_path = getattr(trainer, "final_checkpoint_path", None)

    if full_ckpt_path is None and not analysis_enabled:
        if rank == 0:
            logger.info(
                "Skipping final full-state gather: checkpointing and analysis "
                "are both disabled."
            )
        return ""

    if full_ckpt_path is None:
        fsdp_model = trainer.fsdp_model
        full_ckpt_path = os.path.join(
            run_save_path or "/tmp", "checkpoints", "final_full.pt"
        )
        if rank == 0:
            os.makedirs(os.path.dirname(full_ckpt_path), exist_ok=True)
        completed_epoch = int(
            getattr(trainer, "last_completed_epoch", trainer.epochs - 1)
        )
        save_fsdp_checkpoint(
            fsdp_model,
            trainer.optimizer,
            completed_epoch,
            full_ckpt_path,
            rank=rank,
        )
        if rank == 0:
            logger.info(f"Full checkpoint saved to {full_ckpt_path}")
    elif rank == 0:
        logger.info("Reusing trainer final checkpoint at %s", full_ckpt_path)

    if rank == 0 and analysis_enabled and analysis_manager is not None:
        _run_final_analysis_on_plain_model(
            config=config,
            analysis_manager=analysis_manager,
            full_ckpt_path=full_ckpt_path,
            encoder_network=encoder_network,
            local_rank=local_rank,
        )

    return full_ckpt_path


def _save_rank0_training_results(
    *,
    rank: int,
    results: dict,
    run_save_path,
) -> None:
    """Log and save final training results on rank 0."""
    if rank != 0:
        return

    logger.info("Training completed successfully!")
    logger.info(f"Final training loss: {results.get('final_train_loss', 'N/A')}")
    logger.info(f"Final validation loss: {results.get('final_valid_loss', 'N/A')}")
    save_dict(results, run_save_path, "training_results.json")


def _encode_rank0_analysis_datasets(
    *,
    model,
    config,
    train_ds,
    valid_ds,
    test_ds,
):
    """Encode datasets for rank-0 analysis without running FSDP autoencoder pretraining."""
    from dendritic_modeling.networks.architectures.classical.autoencoder import (
        BaseAutoencoder,
    )

    encoder_network = model.encoder_network
    if isinstance(encoder_network, BaseAutoencoder):
        logger.warning(
            "BaseAutoencoder pretraining is not supported inside the "
            "FSDP script.  Pretrain the encoder separately, then load "
            "its checkpoint.  Skipping encoder pretraining."
        )
        return encoder_network, train_ds, valid_ds, test_ds

    logger.info("Encoding datasets for analysis...")
    return load_train_encoder_network(
        encoder_network=encoder_network,
        encoder_train_config=config.training.encoder,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        encoder_network_config=config.model.encoder,
        task_config=config.data,
    )


def _prepare_rank_analysis_datasets(
    *,
    rank: int,
    model,
    config,
    train_ds,
    valid_ds,
    test_ds,
):
    """Return encoded analysis datasets on rank 0 and placeholders elsewhere."""
    if rank == 0:
        return _encode_rank0_analysis_datasets(
            model=model,
            config=config,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
        )

    return Identity(1), None, None, None


class _FSDPLoaderTuningDataset(Dataset):
    """Placeholder used only for backwards-compatible helper calls."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        del index
        return torch.empty(0), torch.empty(0, dtype=torch.long)


def _dataloader_kwargs_from_common(
    common_config,
    dataset: Dataset | None = None,
    *,
    device: torch.device | str | None = None,
) -> dict:
    """Build validated DataLoader kwargs from training.main.common."""
    return _dataloader_tuning_from_common(
        common_config,
        dataset,
        device=device,
    ).as_kwargs()


def _dataloader_tuning_from_common(
    common_config,
    dataset: Dataset | None = None,
    *,
    device: torch.device | str | None = None,
) -> DataLoaderTuning:
    """Resolve DataLoader tuning from training.main.common."""
    default_num_workers = None
    if dataset is None and getattr(common_config, "num_workers", None) is None:
        # Preserve the historical standalone helper default.  Real call sites
        # pass a dataset so shared tuning can choose tensor-vs-file loading.
        default_num_workers = min(4, visible_cpu_count())

    return resolve_dataloader_tuning_from_config(
        dataset if dataset is not None else _FSDPLoaderTuningDataset(),
        common_config,
        device=device,
        default_num_workers=default_num_workers,
    )


def setup_data_loaders(config, rank, world_size):
    """Setup distributed data loaders."""
    resolved_seeds = resolve_experiment_seeds(config.experiment, write_back=True)
    set_seed(
        resolved_seeds.dataset_seed,
        deterministic=bool(getattr(config.experiment, "deterministic", True)),
        cudnn_benchmark=bool(getattr(config.experiment, "cudnn_benchmark", False)),
        allow_tf32=bool(getattr(config.experiment, "allow_tf32", False)),
        float32_matmul_precision=str(
            getattr(config.experiment, "float32_matmul_precision", "highest")
        ),
        strict_deterministic=resolved_seeds.strict_deterministic,
    )
    data_config = config.data
    base_dir = data_config.base_dir or ""
    dataset_specific_params = _get_dataset_specific_params(data_config)
    processing_params = _config_to_plain_dict(getattr(data_config, "processing", None))

    task_cfg = type(
        "TaskConfig",
        (),
        {
            "dataset": data_config.dataset_name,
            "data_path": base_dir or None,
            "train_valid_split": config.experiment.train_valid_split,
            "parameters": {
                **processing_params,
                **dataset_specific_params,
                "label_noise_seed": resolved_seeds.dataset_seed,
                "split_seed": resolved_seeds.split_seed,
            },
        },
    )()
    task_cfg.parameters.setdefault("seed", resolved_seeds.dataset_seed)

    train_ds, valid_ds, test_ds = get_unified_datasets(task_cfg=task_cfg)
    _validate_sealed_test_dataset(config.experiment, test_ds)

    # Create distributed sampler for training only.
    # Validation uses a non-padded sampler: each rank gets a disjoint
    # shard *without* padding, and the trainer reduces exact counts
    # (loss_sum, correct, total_examples) via all-reduce SUM.
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=resolved_seeds.loader_seed,
    )

    # PaddedDistributedEvalSampler gives each rank an equal number of
    # indices (required by FSDP all-gather) but marks the extras as
    # padding (-1 sentinels).  The trainer's _epoch_valid_fsdp excludes
    # padding from the reduction, so validation is numerically identical
    # to a non-distributed run over the full dataset.
    valid_sampler = PaddedDistributedEvalSampler(
        valid_ds,
        num_replicas=world_size,
        rank=rank,
    )

    # Create data loaders
    batch_size = config.training.main.common.batch_size
    loader_tuning = _dataloader_tuning_from_common(
        config.training.main.common,
        train_ds,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if rank == 0:
        logger.info("FSDP DataLoader tuning: %s", loader_tuning.summary(train_ds))
    loader_kwargs = loader_tuning.as_kwargs()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        **seeded_dataloader_kwargs(
            resolved_seeds.loader_seed,
            stream=0,
            rank=rank,
        ),
        **loader_kwargs,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        sampler=valid_sampler,
        **seeded_dataloader_kwargs(
            resolved_seeds.loader_seed,
            stream=1,
            rank=rank,
        ),
        **loader_kwargs,
    )

    return train_loader, valid_loader, test_ds, train_ds, valid_ds


def main(config_path):
    """Main FSDP training function."""
    script_started = time.perf_counter()
    # Setup distributed environment
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        logger.info(f"Starting FSDP training on {world_size} GPUs")

    # Load config
    config = load_config(config_path)
    from dendritic_modeling.scripts.training.train_experiments import (
        _validate_sealed_test_analysis,
    )

    _validate_sealed_test_analysis(config.experiment, config.analysis)

    resolved_seeds = resolve_experiment_seeds(config.experiment, write_back=True)
    _apply_topology_seed_defaults(config.model, resolved_seeds.topology_seed)
    _apply_initialization_seed_defaults(config.model, resolved_seeds.model_seed)
    set_seed(
        resolved_seeds.model_seed,
        deterministic=bool(getattr(config.experiment, "deterministic", True)),
        cudnn_benchmark=bool(getattr(config.experiment, "cudnn_benchmark", False)),
        allow_tf32=bool(getattr(config.experiment, "allow_tf32", False)),
        float32_matmul_precision=str(
            getattr(config.experiment, "float32_matmul_precision", "highest")
        ),
        strict_deterministic=resolved_seeds.strict_deterministic,
    )
    if rank == 0:
        logger.info("Resolved RNG seeds: %s", resolved_seeds.asdict())

    # Setup output directory (only on rank 0)
    if rank == 0:
        run_save_path = _setup_rank0_run_directory(config)
        save_dict(resolved_seeds.asdict(), run_save_path, "resolved_seeds.json")
    else:
        run_save_path = None

    # Setup WandB (only rank 0)
    wandb_run = setup_wandb(config, rank, world_size) if rank == 0 else None

    # Create model
    model_creation_started = time.perf_counter()
    model = create_large_model(config)
    _apply_fsdp_trainability_policy(model, config.distributed.fsdp)
    model_creation_seconds = time.perf_counter() - model_creation_started

    # Setup data loaders
    data_setup_started = time.perf_counter()
    train_loader, valid_loader, test_ds, train_ds, valid_ds = setup_data_loaders(
        config, rank, world_size
    )
    data_setup_seconds = time.perf_counter() - data_setup_started

    # Encode datasets for analysis (rank 0 only).
    # Note: FSDP does NOT support autoencoder pretraining.  If the
    # encoder is a BaseAutoencoder, create_large_model() already
    # collapsed it to its encoder sub-module, so
    # load_train_encoder_network() will return immediately.
    # To pretrain an autoencoder encoder, run it *before* FSDP
    # training with the standard single-GPU script, then point
    # the config at the saved checkpoint.
    analysis_enabled = config.analysis.any_enabled()
    if analysis_enabled:
        encoder_network, encoded_train_ds, encoded_valid_ds, encoded_test_ds = (
            _prepare_rank_analysis_datasets(
                rank=rank,
                model=model,
                config=config,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
            )
        )
    else:
        encoder_network = Identity(1)
        encoded_train_ds = None
        encoded_valid_ds = None
        encoded_test_ds = None
        if rank == 0:
            logger.info(
                "No analysis modules enabled; skipping analysis dataset encoding."
            )

    # Generate detailed parameter summary with real sample input
    if rank == 0:  # Only on rank 0 to avoid duplicate output
        _print_rank0_model_summary(model, train_ds)

    train_common, fsdp_config, optimizer_config = _build_fsdp_runtime_configs(config)

    # Setup analysis manager (only on rank 0)
    analysis_manager = None
    if rank == 0 and analysis_enabled:
        analysis_manager = _setup_rank0_analysis_manager(
            config=config,
            model=model,
            encoder_network=encoder_network,
            encoded_train_ds=encoded_train_ds,
            encoded_valid_ds=encoded_valid_ds,
            encoded_test_ds=encoded_test_ds,
            run_save_path=run_save_path,
            local_rank=local_rank,
        )
    elif rank == 0:
        logger.info("No final analysis modules enabled; skipping AnalysisManager.")

    trainer = _create_fsdp_trainer(
        rank=rank,
        world_size=world_size,
        fsdp_config=fsdp_config,
        run_save_path=run_save_path,
        train_common=train_common,
        optimizer_config=optimizer_config,
        analysis_manager=analysis_manager,
        wandb_run=wandb_run,
        config=config,
    )

    if rank == 0:
        logger.info("Starting FSDP training...")
        logger.info(
            f"Plot losses enabled: {getattr(train_common, 'plot_losses', False)}"
        )
        logger.info(f"Save path for plots: {run_save_path}")

    # Train the model
    results = trainer.train(model, train_loader, valid_loader)
    results.update(
        {
            "model_creation_seconds": model_creation_seconds,
            "data_setup_seconds": data_setup_seconds,
            "runtime_through_training_seconds": time.perf_counter() - script_started,
            "world_size": world_size,
        }
    )

    _save_rank0_training_results(
        rank=rank,
        results=results,
        run_save_path=run_save_path,
    )

    # --- Final analysis: gather full state dict, reload into plain model ---
    # FSDP shards parameters across ranks; analysis on the sharded model
    # produces incorrect results.  The safe pattern is:
    # 1. Save a full (unsharded) checkpoint from the FSDP model (all ranks
    #    participate in the all-gather, rank 0 saves).
    # 2. On rank 0, create a fresh non-FSDP model and load that checkpoint.
    # 3. Run analysis on the plain model.
    _save_final_fsdp_checkpoint_and_run_analysis(
        rank=rank,
        trainer=trainer,
        run_save_path=run_save_path,
        config=config,
        analysis_manager=analysis_manager,
        encoder_network=encoder_network,
        local_rank=local_rank,
    )

    # Clean up distributed
    dist.destroy_process_group()

    if rank == 0:
        logger.info("FSDP training script completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FSDP training for large dendritic models"
    )
    parser.add_argument("config", help="Path to the configuration YAML file")
    args = parser.parse_args()

    main(args.config)
