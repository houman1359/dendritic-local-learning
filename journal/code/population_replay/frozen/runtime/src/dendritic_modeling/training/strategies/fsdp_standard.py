"""
FSDP (Fully Sharded Data Parallel) training strategy.

This module provides FSDP-based distributed training functionality
for scaling to very large models across multiple GPUs and nodes.
"""

import logging
import os
import time
from collections.abc import Iterable, Iterator
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:  # PyTorch exposes the FSDP-aware scaler from this submodule.
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
except Exception:  # pragma: no cover - depends on installed PyTorch version.
    ShardedGradScaler = None

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.distributed_samplers import (
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.training.fsdp_utils import (
    get_fsdp_config,
    save_fsdp_checkpoint,
    wrap_model_with_fsdp,
)
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.utils.hooks import iter_modules_matching

logger = logging.getLogger(__name__)

_TOPK_PARAM_GROUP_CLASS_NAMES = (
    "TopKLinear",
    "IndexedSparseLinear",
    "IndexedDynamicTopKLinear",
    "IndexedRewireLinear",
)


def _optimizer_config_value(
    optimizer_config: Optional[dict], key: str, default: float
) -> Any:
    return (optimizer_config or {}).get(key, default)


def _sequence_lengths_from_batch(
    batch,
    *,
    x_batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    """Return optional per-sample sequence lengths from a recurrent batch.

    Sequence datasets use ``(inputs, labels, lengths)``.  Image and tabular
    datasets may also expose extra metadata, so only accept a one-value-per-row
    tensor when the input has an explicit time dimension.
    """
    if x_batch.dim() != 3 or not isinstance(batch, (tuple, list)) or len(batch) < 3:
        return None
    lengths = batch[2]
    if not torch.is_tensor(lengths):
        try:
            lengths = torch.as_tensor(lengths)
        except (TypeError, ValueError):
            return None
    lengths = lengths.reshape(-1)
    if lengths.numel() != x_batch.shape[0]:
        return None
    return lengths.to(device=device, dtype=torch.long, non_blocking=True)


def _tensor_diagnostics(tensor: torch.Tensor) -> dict[str, float | int]:
    """Return compact, JSON-safe finite-value diagnostics for a tensor."""
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    diagnostics: dict[str, float | int] = {
        "numel": values.numel(),
        "finite_count": finite_count,
    }
    if finite_count == 0:
        return diagnostics
    finite_values = values[finite]
    diagnostics.update(
        {
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "rms": float(finite_values.square().mean().sqrt().item()),
        }
    )
    if finite_count == values.numel() and values.ndim >= 2:
        rows = values.reshape(-1, values.shape[-1])
        diagnostics.update(
            {
                "within_row_std_mean": float(
                    rows.std(dim=-1, unbiased=False).mean().item()
                ),
                "row_mean_std": float(rows.mean(dim=-1).std(unbiased=False).item()),
                "across_row_feature_std_mean": float(
                    rows.std(dim=0, unbiased=False).mean().item()
                ),
            }
        )
    return diagnostics


def _collect_unique_parameters(
    parameters: Iterable[torch.nn.Parameter],
    used_param_ids: set[int],
) -> list[torch.nn.Parameter]:
    unique_params = []
    for param in parameters:
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id not in used_param_ids:
            unique_params.append(param)
            used_param_ids.add(param_id)
    return unique_params


def _iter_modules_by_class_name(
    model: torch.nn.Module,
    class_name: str,
) -> Iterator[torch.nn.Module]:
    """Yield modules whose concrete class name matches ``class_name``."""
    yield from iter_modules_matching(
        model,
        lambda module: module.__class__.__name__ == class_name,
    )


def _collect_module_parameters_by_class_name(
    model: torch.nn.Module,
    class_name: str,
    used_param_ids: set[int],
) -> list[torch.nn.Parameter]:
    params = []
    for module in _iter_modules_by_class_name(model, class_name):
        params.extend(_collect_unique_parameters(module.parameters(), used_param_ids))
    return params


def _reactivation_parameter_candidates(
    module: torch.nn.Module,
) -> tuple[torch.nn.Parameter, ...]:
    if hasattr(module, "m") and hasattr(module, "b"):
        return module.m, module.b
    if hasattr(module, "log_m") and hasattr(module, "b"):
        return module.log_m, module.b
    return ()


def _has_reactivation_parameter_candidates(module: torch.nn.Module) -> bool:
    """Return whether a module exposes trainable reactivation parameters."""
    return (hasattr(module, "m") or hasattr(module, "log_m")) and hasattr(module, "b")


def _collect_reactivation_parameters(
    model: torch.nn.Module,
    used_param_ids: set[int],
) -> list[torch.nn.Parameter]:
    params = []
    for candidates in _iter_reactivation_parameter_candidates(model):
        params.extend(_collect_unique_parameters(candidates, used_param_ids))
    return params


def _first_linear_weight(module: torch.nn.Module) -> torch.nn.Parameter | None:
    """Return the first Linear weight in module traversal order."""
    for child in iter_modules_matching(
        module,
        lambda item: isinstance(item, torch.nn.Linear),
    ):
        return child.weight
    return None


def _iter_reactivation_parameter_candidates(
    model: torch.nn.Module,
) -> Iterator[tuple[torch.nn.Parameter, ...]]:
    """Yield non-empty reactivation parameter candidate tuples in module order."""
    for module in iter_modules_matching(model, _has_reactivation_parameter_candidates):
        yield _reactivation_parameter_candidates(module)


def _append_fsdp_param_group(
    param_groups: list[dict[str, Any]],
    params: list[torch.nn.Parameter],
    *,
    name: str,
    optimizer_config: Optional[dict],
    lr_key: str,
    default_lr: float,
) -> None:
    if params:
        param_groups.append(
            {
                "name": name,
                "params": params,
                "lr": _optimizer_config_value(optimizer_config, lr_key, default_lr),
            }
        )


class FSDPTrainer(Trainer):
    """
    FSDP-enabled trainer for large dendritic models.

    This trainer extends the standard trainer with FSDP support,
    allowing training of models that don't fit on a single GPU.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        fsdp_config: Optional[dict] = None,
        checkpoint_dir: Optional[str] = None,
        save_every_n_epochs: Optional[int] = None,
        optimizer_config: Optional[dict] = None,
        analysis_manager: Optional[object] = None,
        wandb_run: Optional[object] = None,
        log_first_step_diagnostics: bool = False,
        cache_frozen_core_outputs: bool = False,
        cached_core_batch_size: int | None = None,
        **kwargs,
    ):
        """
        Initialize FSDP trainer.

        Args:
            rank: Global rank of current process
            world_size: Total number of processes
            fsdp_config: FSDP configuration
            checkpoint_dir: Directory to save checkpoints
            save_every_n_epochs: Save checkpoint every N epochs (None disables periodic checkpointing)
            optimizer_config: Optimizer configuration
            **kwargs: Arguments passed to base Trainer
        """
        # Store configuration
        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.fsdp_config = fsdp_config or get_fsdp_config()
        self.save_every_n_epochs = save_every_n_epochs

        # Set up checkpoint directory structure (consistent with standard trainer)
        if checkpoint_dir is not None:
            self.checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
        else:
            self.checkpoint_dir = f"results/fsdp_run_{time.time()}/checkpoints"
        self.optimizer_config = optimizer_config or {}
        self.analysis_manager = analysis_manager
        self.wandb_run = wandb_run
        self.log_first_step_diagnostics = bool(log_first_step_diagnostics)
        self.cache_frozen_core_outputs = bool(cache_frozen_core_outputs)
        if cached_core_batch_size is not None and int(cached_core_batch_size) <= 0:
            raise ValueError("cached_core_batch_size must be positive when provided")
        self.cached_core_batch_size = (
            int(cached_core_batch_size) if cached_core_batch_size is not None else None
        )
        self.first_step_diagnostics: dict[str, Any] | None = None
        self._decoder_input_diagnostics: dict[str, float | int] | None = None
        self._use_precomputed_core = False

        # Set device using local rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

        # Initialize parent with dummy optimizer (we'll create the real one after FSDP wrapping)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.01)
        if self.rank == 0:
            logger.info(f"Initializing FSDPTrainer with kwargs: {kwargs}")

        # Store checkpoint_dir before parent call (parent might override it)
        checkpoint_dir_backup = self.checkpoint_dir

        fsdp_kwargs = kwargs.copy()
        super().__init__(optimizer=dummy_optimizer, **fsdp_kwargs)
        # Restore checkpoint_dir after parent call
        self.checkpoint_dir = checkpoint_dir_backup

        # Create checkpoint directory
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize loss function based on loss_function parameter
        loss_fn_name = kwargs.get("loss_function", "cat_nll")
        if loss_fn_name in ["cat_nll", "cross_entropy", "ce"]:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_fn_name == "mse":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn_name}")

        # Mixed Precision Training Support
        self.use_amp = kwargs.get("use_amp", False)
        if self.use_amp:
            try:
                # autocast and GradScaler already imported at top
                scaler_cls = ShardedGradScaler or GradScaler
                self.scaler = scaler_cls()
                if self.rank == 0:
                    logger.info("Mixed precision training enabled for FSDP")
            except ImportError:
                if self.rank == 0:
                    logger.warning(
                        "AMP not available, falling back to standard precision"
                    )
                self.use_amp = False

    def _wrap_model(self, model: BaseModel) -> BaseModel:
        """Wrap model with FSDP."""
        # cpu_init: do NOT move the model to the GPU before wrapping. Doing so
        # materializes the full unsharded model on every rank (37.6 GB at 1M --
        # the measured cause of the 1M gate's OOM inside FSDP.__init__).
        # FSDP's device_id machinery moves each wrapped unit's states itself,
        # so the CPU-resident model reaches the GPU shard-by-shard. Whatever
        # FSDP does not manage (readout head, encoder under 10M params) is
        # moved explicitly afterwards -- child-wise, skipping any subtree that
        # contains an FSDP instance, so sharded storage is never re-touched.
        cpu_init = bool((self.fsdp_config or {}).get("_cpu_init", False))
        if not cpu_init:
            # Historical behavior: move base model to device first.
            model = model.to(self.device)

        # Wrap with FSDP
        if hasattr(model, "core_network"):
            # For models with separate encoder/decoder
            model.core_network = wrap_model_with_fsdp(
                model.core_network, self.fsdp_config
            )
        elif hasattr(model, "net"):
            # For models with separate encoder/decoder
            model.net = wrap_model_with_fsdp(model.net, self.fsdp_config)

            # Optionally wrap encoder/decoder if they're large
            if hasattr(model, "encoder_network"):
                param_count = sum(p.numel() for p in model.encoder_network.parameters())
                if param_count > 10_000_000:  # > 10M parameters
                    model.encoder_network = wrap_model_with_fsdp(
                        model.encoder_network, self.fsdp_config
                    )

            if hasattr(model, "decoder_network"):
                param_count = sum(p.numel() for p in model.decoder_network.parameters())
                if param_count > 10_000_000:  # > 10M parameters
                    model.decoder_network = wrap_model_with_fsdp(
                        model.decoder_network, self.fsdp_config
                    )
        else:
            # Wrap entire model
            model = wrap_model_with_fsdp(model, self.fsdp_config)

        if cpu_init:
            for child in model.children():
                if not any(
                    isinstance(m, FullyShardedDataParallel) for m in child.modules()
                ):
                    child.to(self.device)
            # Root-level tensors owned directly by the wrapper module, if any.
            for tensor in list(model.parameters(recurse=False)) + list(
                model.buffers(recurse=False)
            ):
                tensor.data = tensor.data.to(self.device)

        return model

    def create_optimizer(self, model: BaseModel) -> torch.optim.Optimizer:
        """Create optimizer with FSDP-aware parameter groups."""
        # Exclude frozen parameters so readout-only runs do not allocate Adam
        # state for the contact-heavy recurrent core.
        all_params = [param for param in model.parameters() if param.requires_grad]
        if not all_params:
            raise ValueError("FSDP training requires at least one trainable parameter")
        optimizer_config = self.optimizer_config or {}

        # If not using split params, use all parameters with single LR
        if not optimizer_config.get("split_params", False):
            return torch.optim.Adam(
                all_params,
                lr=_optimizer_config_value(optimizer_config, "lr", 0.001),
                weight_decay=_optimizer_config_value(
                    optimizer_config, "weight_decay", 0.0
                ),
            )

        # Otherwise, create parameter groups
        param_groups = []
        used_params = set()

        topk_params = []
        for class_name in _TOPK_PARAM_GROUP_CLASS_NAMES:
            topk_params.extend(
                _collect_module_parameters_by_class_name(
                    model,
                    class_name,
                    used_params,
                )
            )
        _append_fsdp_param_group(
            param_groups,
            topk_params,
            name="topk",
            optimizer_config=optimizer_config,
            lr_key="topk_lr",
            default_lr=0.001,
        )

        blocklinear_params = _collect_module_parameters_by_class_name(
            model, "BlockLinear", used_params
        )
        _append_fsdp_param_group(
            param_groups,
            blocklinear_params,
            name="blocklinear",
            optimizer_config=optimizer_config,
            lr_key="blocklinear_lr",
            default_lr=0.001,
        )

        reactivation_params = _collect_reactivation_parameters(model, used_params)
        _append_fsdp_param_group(
            param_groups,
            reactivation_params,
            name="reactivation",
            optimizer_config=optimizer_config,
            lr_key="reactivation_lr",
            default_lr=0.001,
        )

        if hasattr(model, "decoder_network"):
            decoder_input_lr = optimizer_config.get("decoder_input_lr")
            if decoder_input_lr is not None:
                first_weight = _first_linear_weight(model.decoder_network)
                decoder_input_params = (
                    _collect_unique_parameters([first_weight], used_params)
                    if first_weight is not None
                    else []
                )
                _append_fsdp_param_group(
                    param_groups,
                    decoder_input_params,
                    name="decoder_input",
                    optimizer_config=optimizer_config,
                    lr_key="decoder_input_lr",
                    default_lr=0.001,
                )
            decoder_params = _collect_unique_parameters(
                model.decoder_network.parameters(), used_params
            )
        else:
            decoder_params = []
        _append_fsdp_param_group(
            param_groups,
            decoder_params,
            name="decoder",
            optimizer_config=optimizer_config,
            lr_key="decoder_lr",
            default_lr=0.001,
        )

        # All other parameters
        other_params = [p for p in all_params if id(p) not in used_params]
        _append_fsdp_param_group(
            param_groups,
            other_params,
            name="other",
            optimizer_config=optimizer_config,
            lr_key="lr",
            default_lr=0.001,
        )

        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=_optimizer_config_value(optimizer_config, "weight_decay", 0.0),
        )
        param_counts = [
            sum(param.numel() for param in group["params"])
            for group in optimizer.param_groups
        ]
        if dist.is_available() and dist.is_initialized():
            count_device = all_params[0].device
            global_counts = torch.tensor(
                param_counts,
                dtype=torch.int64,
                device=count_device,
            )
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
            param_counts = [int(value) for value in global_counts.tolist()]
        if getattr(self, "rank", 0) == 0:
            group_summary = ", ".join(
                (
                    f"{group['name']}: lr={group['lr']:.3g}, "
                    f"global_params={param_count:,}"
                )
                for group, param_count in zip(
                    optimizer.param_groups,
                    param_counts,
                    strict=True,
                )
            )
            logger.info("FSDP optimizer groups: %s", group_summary)
        return optimizer

    def _cache_core_output_loader(
        self,
        model: BaseModel,
        loader: DataLoader,
        *,
        shuffle: bool,
        seed_offset: int,
    ) -> DataLoader:
        """Materialize one frozen-core output for each sample on this rank."""
        decoder = getattr(model, "decoder_network", None)
        if decoder is None:
            raise ValueError("cache_frozen_core_outputs requires model.decoder_network")

        cached_features: list[torch.Tensor] = []
        cached_targets: list[torch.Tensor] = []

        def capture_core_output(_module, args) -> None:
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError("Decoder did not receive a tensor core output")
            cached_features.append(args[0].detach().to(device="cpu", copy=True))

        hook = decoder.register_forward_pre_hook(capture_core_output)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                        raise ValueError(
                            "Cached frozen-core training requires (input, target) batches"
                        )
                    x_batch = batch[0].to(self.device)
                    seq_lengths = _sequence_lengths_from_batch(
                        batch,
                        x_batch=x_batch,
                        device=self.device,
                    )
                    captures_before = len(cached_features)
                    model(x_batch, seq_lengths=seq_lengths)
                    if len(cached_features) != captures_before + 1:
                        raise RuntimeError(
                            "Expected exactly one decoder input per cached batch"
                        )
                    cached_targets.append(batch[1].detach().to(device="cpu", copy=True))
        finally:
            hook.remove()
            model.train(was_training)

        if not cached_features:
            raise ValueError("Cannot cache frozen-core outputs from an empty loader")
        features = torch.cat(cached_features, dim=0)
        targets = torch.cat(cached_targets, dim=0)
        if features.shape[0] != targets.shape[0]:
            raise RuntimeError(
                "Cached core-output and target counts differ: "
                f"{features.shape[0]} != {targets.shape[0]}"
            )

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.rank + seed_offset)
        cached_batch_size = self.cached_core_batch_size or loader.batch_size
        if cached_batch_size is None:
            raise ValueError(
                "cached_core_batch_size is required when the source loader "
                "does not define batch_size"
            )
        cached_loader = DataLoader(
            TensorDataset(features, targets),
            batch_size=cached_batch_size,
            shuffle=shuffle,
            drop_last=loader.drop_last,
            num_workers=0,
            pin_memory=loader.pin_memory,
            generator=generator,
        )
        if isinstance(loader.sampler, PaddedDistributedEvalSampler):
            cached_loader._dendritic_real_count = len(loader.sampler.real_indices)
        return cached_loader

    def _cache_frozen_core_loaders(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
    ) -> tuple[DataLoader, DataLoader | None]:
        """Replace raw loaders with rank-local deterministic core features."""
        cached_train_loader = self._cache_core_output_loader(
            model,
            train_loader,
            shuffle=True,
            seed_offset=17,
        )
        cached_valid_loader = (
            self._cache_core_output_loader(
                model,
                valid_loader,
                shuffle=False,
                seed_offset=29,
            )
            if valid_loader is not None
            else None
        )
        local_counts = torch.tensor(
            [
                len(cached_train_loader.dataset),
                (
                    len(cached_valid_loader.dataset)
                    if cached_valid_loader is not None
                    else 0
                ),
            ],
            dtype=torch.int64,
            device=self.device,
        )
        dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            feature_shape = cached_train_loader.dataset.tensors[0].shape[1:]
            logger.info(
                "Cached frozen-core outputs: train=%s, valid=%s, feature_shape=%s",
                f"{int(local_counts[0].item()):,}",
                f"{int(local_counts[1].item()):,}",
                tuple(feature_shape),
            )
        self._use_precomputed_core = True
        return cached_train_loader, cached_valid_loader

    def train(
        self,
        model: BaseModel,
        train_loader: DataLoader | Dataset,
        valid_loader: Optional[DataLoader | Dataset] = None,
    ) -> dict[str, Any]:
        """
        Train model using FSDP with pre-created DataLoaders.

        Args:
            model: Model to train (will be wrapped with FSDP)
            train_loader: Training DataLoader (or training Dataset for auto-wrapping)
            valid_loader: Validation DataLoader (or validation Dataset for auto-wrapping)

        Returns:
            Training results dictionary
        """
        trainer_started = time.perf_counter()
        if isinstance(train_loader, DataLoader):
            if valid_loader is not None and not isinstance(valid_loader, DataLoader):
                raise TypeError(
                    "If train_loader is a DataLoader, valid_loader must also be a DataLoader or None."
                )
        else:
            train_loader, valid_loader = self._dataloaders(train_loader, valid_loader)

        # Include FSDP wrapping, state synchronization, training, validation,
        # and checkpoint work in one per-rank peak-memory measurement.
        track_cuda_memory = torch.cuda.is_available()
        if track_cuda_memory:
            torch.cuda.reset_peak_memory_stats(self.device)

        # Wrap model with FSDP
        model = wrap_model_with_fsdp(model, self.fsdp_config)

        # Only move to device if CPU offloading is NOT enabled
        # When CPU offloading is enabled, FSDP manages device placement
        cpu_offload_enabled = self.fsdp_config.get(
            "cpu_offload"
        ) is not None and self.fsdp_config.get("cpu_offload")
        if not cpu_offload_enabled:
            model = model.to(self.device)

        # Store the FSDP-wrapped model so callers can use it for
        # checkpoint saving / full-param gathering after training.
        self.fsdp_model = model

        # Create optimizer after FSDP wrapping
        self.optimizer = self.create_optimizer(model)

        diagnostic_hook = None
        if self.log_first_step_diagnostics and hasattr(model, "decoder_network"):

            def capture_decoder_input(_module, args) -> None:
                if self._decoder_input_diagnostics is not None or not args:
                    return
                decoder_input = args[0]
                if torch.is_tensor(decoder_input):
                    self._decoder_input_diagnostics = _tensor_diagnostics(decoder_input)

            diagnostic_hook = model.decoder_network.register_forward_pre_hook(
                capture_decoder_input
            )

        fsdp_setup_peak_memory = None
        if track_cuda_memory:
            torch.cuda.synchronize(self.device)
            fsdp_setup_peak_memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(self.device),
                    torch.cuda.max_memory_reserved(self.device),
                ],
                dtype=torch.int64,
                device=self.device,
            )
            dist.all_reduce(fsdp_setup_peak_memory, op=dist.ReduceOp.MAX)
            # Measure the recurrent train/validation phase independently while
            # retaining the setup peak above for an overall maximum.
            torch.cuda.reset_peak_memory_stats(self.device)
        fsdp_setup_seconds = time.perf_counter() - trainer_started

        frozen_core_cache_seconds = None
        if self.cache_frozen_core_outputs:
            if not bool(getattr(model, "_freeze_core_forward", False)):
                raise ValueError(
                    "cache_frozen_core_outputs requires the frozen-core "
                    "trainability policy"
                )
            if track_cuda_memory:
                torch.cuda.synchronize(self.device)
            cache_started = time.perf_counter()
            train_loader, valid_loader = self._cache_frozen_core_loaders(
                model,
                train_loader,
                valid_loader,
            )
            if track_cuda_memory:
                torch.cuda.synchronize(self.device)
            frozen_core_cache_seconds = time.perf_counter() - cache_started

        # Training loop
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        epoch_times_seconds = []
        best_valid_loss = float("inf")
        patience_counter = 0
        self.last_completed_epoch = -1

        if self.rank == 0:
            logger.info(f"Starting training for {self.epochs} epochs")

        training_loop_started = time.perf_counter()
        for epoch in range(self.epochs):
            self.last_completed_epoch = epoch
            if self.rank == 0:
                logger.info(f"Starting epoch {epoch}")
            epoch_start = time.perf_counter()

            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # Training
            train_loss, train_acc = self._epoch_train_fsdp(model, train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            valid_loss = None
            valid_acc = None
            if valid_loader is not None:
                if hasattr(valid_loader.sampler, "set_epoch"):
                    valid_loader.sampler.set_epoch(epoch)
                valid_loss, valid_acc = self._epoch_valid_fsdp(model, valid_loader)
                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)

                improved = valid_loss < best_valid_loss
                if improved:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    if self.early_stopping and self.checkpointing:
                        save_fsdp_checkpoint(
                            model,
                            self.optimizer,
                            epoch,
                            os.path.join(self.checkpoint_dir, "best_model.pt"),
                            rank=self.rank,
                        )
                elif self.early_stopping:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.rank == 0:
                            logger.info(f"Early stopping at epoch {epoch}")
                        break

            # Epoch-level analysis is intentionally skipped for FSDP.
            # FSDP shards model parameters across ranks; running analysis
            # on the sharded model produces incorrect results.  Final
            # analysis runs on rank 0 after gathering the full state dict
            # (see train_experiments_fsdp.py).

            # Logging
            epoch_time = time.perf_counter() - epoch_start
            epoch_times_seconds.append(epoch_time)
            if self.rank == 0 and epoch % self.print_every == 0:
                msg = f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}"
                if valid_loss is not None and valid_acc is not None:
                    msg += (
                        f" - Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_acc:.4f}"
                    )
                msg += f" - Time: {epoch_time:.2f}s"
                logger.info(msg)

            # Checkpointing
            if (
                self.checkpointing
                and self.save_every_n_epochs is not None
                and self.save_every_n_epochs > 0
                and (epoch + 1) % self.save_every_n_epochs == 0
                and (epoch + 1) < self.epochs
            ):
                save_fsdp_checkpoint(
                    model,
                    self.optimizer,
                    epoch,
                    os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
                    rank=self.rank,
                )
        training_loop_seconds = time.perf_counter() - training_loop_started

        # Final save
        self.final_checkpoint_path = None
        if self.checkpointing:
            self.final_checkpoint_path = os.path.join(
                self.checkpoint_dir, "final_model.pt"
            )
            save_fsdp_checkpoint(
                model,
                self.optimizer,
                self.last_completed_epoch,
                self.final_checkpoint_path,
                rank=self.rank,
            )

        # Generate plots (only on rank 0)
        if self.rank == 0:
            self._loss_plotting(train_losses, valid_losses)
            # Removed redundant accuracy plotting - handled by performance option

        results = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "best_valid_loss": best_valid_loss,
            "train_accs": train_accs,
            "valid_accs": valid_accs,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_valid_loss": valid_losses[-1] if valid_losses else None,
            "fsdp_setup_seconds": fsdp_setup_seconds,
            "frozen_core_cache_seconds": frozen_core_cache_seconds,
            "epoch_times_seconds": epoch_times_seconds,
            "training_loop_seconds": training_loop_seconds,
            "trainer_total_seconds": time.perf_counter() - trainer_started,
        }
        if diagnostic_hook is not None:
            diagnostic_hook.remove()
        if self.first_step_diagnostics is not None:
            results["first_step_diagnostics"] = self.first_step_diagnostics
        if track_cuda_memory:
            torch.cuda.synchronize(self.device)
            training_peak_memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(self.device),
                    torch.cuda.max_memory_reserved(self.device),
                ],
                dtype=torch.int64,
                device=self.device,
            )
            dist.all_reduce(training_peak_memory, op=dist.ReduceOp.MAX)
            assert fsdp_setup_peak_memory is not None
            peak_memory = torch.maximum(
                fsdp_setup_peak_memory,
                training_peak_memory,
            )
            results["fsdp_setup_peak_cuda_allocated_bytes"] = int(
                fsdp_setup_peak_memory[0].item()
            )
            results["fsdp_setup_peak_cuda_reserved_bytes"] = int(
                fsdp_setup_peak_memory[1].item()
            )
            results["training_peak_cuda_allocated_bytes"] = int(
                training_peak_memory[0].item()
            )
            results["training_peak_cuda_reserved_bytes"] = int(
                training_peak_memory[1].item()
            )
            results["peak_cuda_allocated_bytes"] = int(peak_memory[0].item())
            results["peak_cuda_reserved_bytes"] = int(peak_memory[1].item())
            if self.rank == 0:
                gib = 1024**3
                logger.info(
                    "Peak CUDA memory across ranks: FSDP setup "
                    "allocated=%.3f/reserved=%.3f GiB; train+validation "
                    "allocated=%.3f/reserved=%.3f GiB; overall "
                    "allocated=%.3f/reserved=%.3f GiB",
                    results["fsdp_setup_peak_cuda_allocated_bytes"] / gib,
                    results["fsdp_setup_peak_cuda_reserved_bytes"] / gib,
                    results["training_peak_cuda_allocated_bytes"] / gib,
                    results["training_peak_cuda_reserved_bytes"] / gib,
                    results["peak_cuda_allocated_bytes"] / gib,
                    results["peak_cuda_reserved_bytes"] / gib,
                )

        return results

    def _epoch_train_fsdp(
        self, model: BaseModel, train_loader: DataLoader
    ) -> tuple[float, float]:
        """Training epoch with FSDP."""
        model.train()
        total_loss = 0
        total_correct = 0
        total_examples = 0

        for _batch_idx, batch in enumerate(train_loader):
            # Clear GPU cache before each forward pass
            torch.cuda.empty_cache()

            x_batch = batch[0].to(self.device)
            y_batch = batch[1].to(self.device)
            seq_lengths = _sequence_lengths_from_batch(
                batch,
                x_batch=x_batch,
                device=self.device,
            )

            if self.use_amp:
                # Mixed precision forward pass
                with autocast():
                    outputs = model(
                        x_batch,
                        seq_lengths=seq_lengths,
                        precomputed_core=self._use_precomputed_core,
                    )
                    loss = self.loss_fn(outputs, y_batch)

                # Backward pass with scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip_value is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = self._clip_grad_norm_fsdp(model)
                else:
                    grad_norm = None

                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                outputs = model(
                    x_batch,
                    seq_lengths=seq_lengths,
                    precomputed_core=self._use_precomputed_core,
                )
                loss = self.loss_fn(outputs, y_batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.grad_clip_value is not None:
                    grad_norm = self._clip_grad_norm_fsdp(model)
                else:
                    grad_norm = None

                # Optimizer step
                self.optimizer.step()

            if self.log_first_step_diagnostics and self.first_step_diagnostics is None:
                self.first_step_diagnostics = {
                    "decoder_input": self._decoder_input_diagnostics,
                    "logits": _tensor_diagnostics(outputs),
                    "loss": float(loss.detach().item()),
                    "grad_norm_before_clip": (
                        float(grad_norm.detach().item())
                        if torch.is_tensor(grad_norm)
                        else (float(grad_norm) if grad_norm is not None else None)
                    ),
                }
                if self.rank == 0:
                    logger.info(
                        "First-step diagnostics: %s",
                        self.first_step_diagnostics,
                    )

            bs = y_batch.size(0)
            total_loss += loss.item() * bs
            total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total_examples += bs

        # All-reduce exact counts across ranks.
        stats = torch.tensor(
            [total_loss, float(total_correct), float(total_examples)],
            device=self.device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        global_examples = stats[2].item()
        avg_loss = stats[0].item() / max(global_examples, 1)
        avg_acc = stats[1].item() / max(global_examples, 1)

        return avg_loss, avg_acc

    def _clip_grad_norm_fsdp(self, model: BaseModel):
        """Clip FSDP gradients through FSDP's collective-aware method."""
        if hasattr(model, "clip_grad_norm_"):
            return model.clip_grad_norm_(self.grad_clip_value)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_value)

    def _epoch_valid_fsdp(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> tuple[float, float]:
        """Validation epoch with FSDP.

        All ranks must participate (FSDP needs all-gather during forward).
        We accumulate sample-weighted loss, correct count, and total
        examples, then all-reduce with SUM for an exact global average.

        When used with ``PaddedDistributedEvalSampler``, padding indices
        (-1) may appear in the last batch.  These are sentinel values that
        index into real data (Python wraps -1 to the last element), so the
        forward pass succeeds on every rank, but we exclude them from the
        stats by tracking the known number of real samples.
        """
        model.eval()
        weighted_loss_sum = 0.0
        total_correct = 0
        total_examples = 0

        # Determine how many real (non-padding) samples this rank has.
        sampler = valid_loader.sampler
        cached_real_count = getattr(
            valid_loader,
            "_dendritic_real_count",
            None,
        )
        if cached_real_count is not None:
            real_count = int(cached_real_count)
        elif isinstance(sampler, PaddedDistributedEvalSampler):
            real_count = len(sampler.real_indices)
        else:
            real_count = None  # unknown — count all

        seen = 0
        with torch.no_grad():
            for batch in valid_loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                seq_lengths = _sequence_lengths_from_batch(
                    batch,
                    x_batch=x_batch,
                    device=self.device,
                )
                bs = y_batch.size(0)

                outputs = model(
                    x_batch,
                    seq_lengths=seq_lengths,
                    precomputed_core=self._use_precomputed_core,
                )
                loss = self.loss_fn(outputs, y_batch)

                if real_count is not None:
                    # Only count the real (non-padding) samples.
                    real_in_batch = min(bs, real_count - seen)
                    if real_in_batch <= 0:
                        # Entire batch is padding — still ran forward for
                        # FSDP all-gather, but skip stats.
                        seen += bs
                        continue
                    if real_in_batch < bs:
                        # Partial padding in this batch.
                        outputs = outputs[:real_in_batch]
                        y_batch = y_batch[:real_in_batch]
                        loss = self.loss_fn(outputs, y_batch)
                        bs = real_in_batch
                    seen += bs

                weighted_loss_sum += loss.item() * bs
                total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                total_examples += bs

        # All-reduce exact counts across ranks.
        stats = torch.tensor(
            [weighted_loss_sum, float(total_correct), float(total_examples)],
            device=self.device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        global_examples = stats[2].item()
        avg_loss = stats[0].item() / max(global_examples, 1)
        avg_acc = stats[1].item() / max(global_examples, 1)

        return avg_loss, avg_acc

    def _reduce_loss(self, loss: float) -> float:
        """Reduce loss across all processes."""
        loss_tensor = torch.tensor(loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor.item()

    def _save_checkpoint(self, model: BaseModel, epoch: int, is_final: bool = False):
        """Save FSDP checkpoint."""
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint_name = (
            "final_checkpoint.pt" if is_final else f"checkpoint_epoch_{epoch}.pt"
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        save_fsdp_checkpoint(
            model=model,
            optimizer=self.optimizer,
            epoch=epoch,
            save_path=checkpoint_path,
            rank=self.rank,
        )

    # Removed _accuracy_plotting method - redundant with performance option plotting
