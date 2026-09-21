"""
Base training strategy classes.

This module contains the base trainer class that provides common functionality
for all training strategies in the dendritic modeling framework.
"""

import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.utils.data
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from dendritic_modeling.config.training import (
    PruningConfig,
    RegularizationConfig,
    ReportingConfig,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ParametricActivation
from dendritic_modeling.plotting import plot_loss_curves
from dendritic_modeling.training.dataloader_utils import (
    resolve_dataloader_tuning_from_config,
    seeded_dataloader_kwargs,
)
from dendritic_modeling.training.distributed_samplers import (
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.training.loss.functions import LossFunction, get_loss_function
from dendritic_modeling.training.regularization import (
    PruningManager,
    RegularizationManager,
)
from dendritic_modeling.utils.hooks import iter_modules_of_type

wandb = None  # wandb is not used in this project (removed 2026-08-20)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ValidationLossStats:
    """Explicit validation aggregates for the non-distributed path.

    ``_epoch_valid`` historically returns the first two fields as a tuple.
    Keeping both batch and sample aggregates here lets checkpoint selection use
    an exact sample-weighted mean without silently changing that tuple's
    second value from a batch count to a sample count.
    """

    batch_loss_sum: float
    batch_count: int
    sample_weighted_loss_sum: float
    sample_count: int

    @property
    def batch_mean(self) -> float:
        return self.batch_loss_sum / max(self.batch_count, 1)

    @property
    def sample_mean(self) -> float:
        return self.sample_weighted_loss_sum / max(self.sample_count, 1)


class Trainer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss_function: str = "mse",
        epochs: int = 100,
        early_stopping: bool = False,
        patience: int = 10,
        batch_size: int = 256,
        shuffle: bool = True,
        grad_clip_value: float = 5,
        checkpointing: bool = False,
        checkpoint_interval: int = 10,
        load_best_state_dict: bool = True,
        plot_losses: bool = False,
        save_path: Optional[str] = None,
        suppress_prints=False,
        print_every=10,
        analysis_manager: Optional[object] = None,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.loss_function: LossFunction = get_loss_function(loss_function)
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.grad_clip_value = grad_clip_value
        self.checkpointing = checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.load_best_state_dict = load_best_state_dict
        self.plot_losses = plot_losses
        self.save_path = save_path
        self.suppress_prints = suppress_prints
        self.print_every = print_every
        self.analysis_manager = analysis_manager
        self.seed = int(kwargs.get("seed", 0) or 0)
        configured_loader_seed = kwargs.get("loader_seed", None)
        self.loader_seed = int(
            self.seed if configured_loader_seed is None else configured_loader_seed
        )

        regularization_config = kwargs.get("regularization_config")
        pruning_config = kwargs.get("pruning_config")
        reporting_config = kwargs.get("reporting_config")

        # Initialize enhanced feature configs with safe defaults
        self.regularization_config = regularization_config or RegularizationConfig()
        self.pruning_config = pruning_config or PruningConfig()
        self.reporting_config = reporting_config or ReportingConfig()

        # Initialize RegularizationManager
        self.regularization_manager = RegularizationManager(self.regularization_config)
        self.pruning_manager = PruningManager(self.pruning_config)

        # Log regularization settings
        if (
            self.regularization_config.l1_weight > 0
            or self.regularization_config.l2_weight > 0
        ):
            logger.info(
                f"L1/L2 regularization enabled: l1_weight={self.regularization_config.l1_weight}, l2_weight={self.regularization_config.l2_weight}"
            )
        if self.regularization_config.split_params and any(
            group_cfg.get("l1_weight", 0) > 0 or group_cfg.get("l2_weight", 0) > 0
            for group_cfg in self.regularization_config.param_group_weights.values()
            if isinstance(group_cfg, dict)
        ):
            logger.info(
                "Split-parameter regularization enabled for groups: %s",
                {
                    name: cfg
                    for name, cfg in self.regularization_config.param_group_weights.items()
                    if isinstance(cfg, dict)
                    and (cfg.get("l1_weight", 0) > 0 or cfg.get("l2_weight", 0) > 0)
                },
            )
        if self.regularization_config.enforce_ei_weight_ratio:
            logger.info(
                f"E/I weight ratio regularization enabled: target_ratio={self.regularization_config.target_ei_weight_ratio}, loss_weight={self.regularization_config.ei_ratio_loss_weight}"
            )

        # Performance optimization flags (disabled by default for speed)
        self.enable_nan_checking = kwargs.get("enable_nan_checking", False)
        self.enable_adaptive_clipping = kwargs.get("enable_adaptive_clipping", False)
        self.profile_dataloader = kwargs.get("profile_dataloader", False)

        self.enable_amp = kwargs.get("use_amp", False)
        if self.enable_amp and torch.cuda.is_available():
            try:
                self.scaler = GradScaler()
                logger.info("Mixed precision training enabled")
            except ImportError:
                logger.warning("AMP not available, falling back to standard precision")
                self.enable_amp = False
        else:
            self.enable_amp = False

        self.enable_profiling = kwargs.get("enable_profiling", False)
        self.profiler = None

        if self.enable_profiling:
            # Auto-disable profiling for small-scale/CPU runs to improve speed
            if not torch.cuda.is_available():
                logger.info("Profiling auto-disabled for CPU run to improve speed")
                self.enable_profiling = False
            elif self.batch_size < 32:
                logger.info(
                    "Profiling auto-disabled for small batch size to reduce overhead"
                )
                self.enable_profiling = False
            else:
                try:
                    import torch.profiler as torch_profiler

                    profiling_output_dir = kwargs.get(
                        "profiling_output_dir", "outputs/profiling/"
                    )
                    os.makedirs(profiling_output_dir, exist_ok=True)

                    self.profiler = torch_profiler.profile(
                        activities=[
                            torch_profiler.ProfilerActivity.CPU,
                            torch_profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch_profiler.schedule(
                            wait=1, warmup=1, active=3, repeat=2
                        ),
                        on_trace_ready=torch_profiler.tensorboard_trace_handler(
                            profiling_output_dir
                        ),
                        record_shapes=True,
                        with_stack=True,
                    )
                    logger.info(
                        f"Profiling enabled - output directory: {profiling_output_dir}"
                    )
                except ImportError:
                    logger.warning("torch.profiler not available, profiling disabled")
                    self.enable_profiling = False

        self.filename_prefix = "standard_"
        self._train_sampler: DistributedSampler | None = None
        self._valid_sampler: Sampler | None = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # DDP helpers
    # ------------------------------------------------------------------

    @property
    def _is_distributed(self) -> bool:
        return dist.is_initialized()

    @property
    def _is_main_process(self) -> bool:
        return not self._is_distributed or dist.get_rank() == 0

    @staticmethod
    def _unwrap_model(model):
        """Return the underlying model, stripping DDP wrapper if present."""
        return model.module if hasattr(model, "module") else model

    def _dataloaders(
        self, train_data: torch.utils.data.Dataset, valid_data: torch.utils.data.Dataset
    ) -> tuple[DataLoader, DataLoader]:
        tuning = resolve_dataloader_tuning_from_config(
            train_data,
            device=getattr(self, "device", "cpu"),
            config=self,
        )
        if not self.suppress_prints and tuning.auto_num_workers:
            total_cpus = os.cpu_count() or tuning.available_cpus
            logger.info(
                "Auto-detected num_workers=%s (visible CPUs: %s, total system CPUs: %s)",
                tuning.num_workers,
                tuning.available_cpus,
                total_cpus,
            )
        if not self.suppress_prints:
            logger.info("DataLoader tuning: %s", tuning.summary(train_data))

        # Use distributed samplers for training and validation under DDP.
        # Validation uses an explicit padding sentinel sampler so every rank
        # executes the same number of forward passes and metrics can be
        # reduced exactly without silent sample duplication.
        if self._is_distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            self._train_sampler = DistributedSampler(
                train_data,
                num_replicas=world_size,
                rank=rank,
                shuffle=self.shuffle,
                seed=self.loader_seed,
            )
            self._valid_sampler = PaddedDistributedEvalSampler(
                valid_data,
                num_replicas=world_size,
                rank=rank,
            )
            train_shuffle = False  # sampler handles shuffling
        else:
            self._train_sampler = None
            self._valid_sampler = None
            train_shuffle = self.shuffle

        loader_rank = dist.get_rank() if self._is_distributed else 0
        train_seed_kwargs = seeded_dataloader_kwargs(
            self.loader_seed,
            stream=0,
            rank=loader_rank,
        )
        valid_seed_kwargs = seeded_dataloader_kwargs(
            self.loader_seed,
            stream=1,
            rank=loader_rank,
        )

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=self._train_sampler,
            **train_seed_kwargs,
            **tuning.as_kwargs(),
        )
        valid_loader = DataLoader(
            valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self._valid_sampler,
            **valid_seed_kwargs,
            **tuning.as_kwargs(),
        )
        return train_loader, valid_loader

    def _initialize_attributes(self, model: BaseModel, total_epochs: int) -> None:
        if not hasattr(self, "device"):
            if self._is_distributed:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self.device = f"cuda:{local_rank}"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(self.device)
        if not hasattr(self, "best_state_dict"):
            self.best_state_dict = deepcopy(model.state_dict())
        if not hasattr(self, "best_loss"):
            self.best_loss = float("inf")
        if not hasattr(self, "best_epoch"):
            self.best_epoch = 1
        if not hasattr(self, "start_time"):
            self.start_time = time.time()
        if not hasattr(self, "logger_info_prefix"):
            self.logger_info_prefix = "[Standard] "
        self.epoch_counter = 0
        self.total_epochs = total_epochs
        self.patience_counter = 0
        self._capture_reactivation_param_states(model)
        self._configure_reactivation_update_mode(model)

    def _iter_reactivation_params(
        self, model: BaseModel
    ) -> list[tuple[torch.nn.Parameter, bool]]:
        """Return reactivation params with their original trainability."""
        params: list[tuple[torch.nn.Parameter, bool]] = []
        for module in iter_modules_of_type(
            self._unwrap_model(model), ParametricActivation
        ):
            for param in module.parameters(recurse=False):
                params.append((param, bool(param.requires_grad)))
        return params

    def _capture_reactivation_param_states(self, model: BaseModel) -> None:
        """Cache the original trainability of reactivation parameters."""
        model_id = id(self._unwrap_model(model))
        if getattr(self, "_reactivation_state_model_id", None) == model_id:
            return
        self._reactivation_state_model_id = model_id
        self._reactivation_param_states = self._iter_reactivation_params(model)

    def _normalize_reactivation_update_mode(self) -> str:
        mode = (
            str(getattr(self, "reactivation_update_mode", "backprop") or "backprop")
            .strip()
            .lower()
        )
        aliases = {
            "bp": "backprop",
            "backprop": "backprop",
            "backpropagation": "backprop",
            "quantile": "quantile",
            "quantile_only": "quantile",
            "quantile-rule": "quantile",
            "quantile_rule": "quantile",
            "frozen": "frozen",
            "freeze": "frozen",
        }
        normalized = aliases.get(mode)
        if normalized is None:
            raise ValueError(
                "Unknown reactivation_update_mode "
                f"{mode!r} (expected 'backprop', 'quantile', or 'frozen')"
            )
        return normalized

    def _set_reactivation_trainability(self, requires_grad: bool) -> None:
        """Apply trainability while preserving originally fixed parameters."""
        for param, original_requires_grad in getattr(
            self, "_reactivation_param_states", []
        ):
            target_requires_grad = original_requires_grad if requires_grad else False
            param.requires_grad_(target_requires_grad)
            if not target_requires_grad:
                param.grad = None

    def _configure_reactivation_update_mode(self, model: BaseModel) -> None:
        """Freeze/unfreeze reactivation params according to the training mode."""
        self._capture_reactivation_param_states(model)
        mode = self._normalize_reactivation_update_mode()
        self.reactivation_update_mode = mode
        if mode == "backprop":
            self._set_reactivation_trainability(True)
            return

        self._set_reactivation_trainability(False)
        if (
            mode == "quantile"
            and getattr(self, "recalibrate_reactivation_every", 0) <= 0
            and self._is_main_process
        ):
            logger.warning(
                "%sreactivation_update_mode='quantile' but "
                "recalibrate_reactivation_every<=0, so (m, b) will stay frozen.",
                getattr(self, "logger_info_prefix", "[Standard] "),
            )

    def _normalize_reactivation_recalibration_mode(self) -> str:
        mode = (
            str(
                getattr(
                    self,
                    "reactivation_recalibration_mode",
                    "from_init_policy",
                )
                or "from_init_policy"
            )
            .strip()
            .lower()
        )
        aliases = {
            "auto": "from_init_policy",
            "from_init_policy": "from_init_policy",
            "init_policy": "from_init_policy",
            "quantile": "occupancy_quantile",
            "occupancy_quantile": "occupancy_quantile",
            "empirical": "median_mad",
            "median_mad": "median_mad",
            "mean_std": "mean_std",
        }
        normalized = aliases.get(mode)
        if normalized is None:
            raise ValueError(
                "Unknown reactivation_recalibration_mode "
                f"{mode!r} (expected 'from_init_policy', 'occupancy_quantile', "
                "'median_mad', or 'mean_std')"
            )
        return normalized

    def _resolve_reactivation_recalibration_mode(
        self, model: BaseModel
    ) -> Optional[str]:
        """Resolve the calibration rule used for periodic (m, b) refreshes."""
        explicit_mode = self._normalize_reactivation_recalibration_mode()
        if explicit_mode != "from_init_policy":
            return explicit_mode

        if self._normalize_reactivation_update_mode() == "quantile":
            return "occupancy_quantile"

        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
            collect_model_data_driven_reactivation_policies,
        )

        requested_policies, saw_layer_policy = (
            collect_model_data_driven_reactivation_policies(self._unwrap_model(model))
        )
        if requested_policies:
            # Defer to per-layer policies so fixed/analytical layers are
            # not recalibrated just because a sibling layer is data-driven.
            return None

        return None if saw_layer_policy else "occupancy_quantile"

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict:
        train_loader, valid_loader = self._dataloaders(train_data, valid_data)

        train_losses = []
        valid_losses = []
        train_losses_base = []
        train_losses_reg = []

        self._initialize_attributes(model, self.epochs)
        self._save_checkpoint(model)

        train_losses, valid_losses, train_losses_base, train_losses_reg = (
            self._run_training_loop(
                n_epochs=self.epochs,
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_losses=train_losses,
                valid_losses=valid_losses,
                train_losses_base=train_losses_base,
                train_losses_reg=train_losses_reg,
                logger_info=self.logger_info_prefix,
                wandb_info={},
            )
        )

        if self._is_main_process:
            logger.info(
                f"Best epoch: {self.best_epoch}, Best loss: {self.best_loss:.4f}"
            )
        if self.load_best_state_dict:
            self._unwrap_model(model).load_state_dict(self.best_state_dict)
            if self._is_main_process:
                logger.info("Final model loaded from best checkpoint")

        results = {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_state_dict": self.best_state_dict,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_losses_base": train_losses_base,
            "train_losses_reg": train_losses_reg,
        }

        self._loss_plotting(
            train_losses, valid_losses, train_losses_base, train_losses_reg
        )

        return results

    def _run_training_loop(
        self,
        n_epochs: int,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: list[float],
        valid_losses: list[float],
        train_losses_base: list[float],
        train_losses_reg: list[float],
        logger_info: str = "",
        wandb_info: dict | None = None,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        if wandb_info is None:
            wandb_info = {}
        for _ in range(1, n_epochs + 1):
            train_losses, valid_losses, train_losses_base, train_losses_reg = (
                self._run_epoch(
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    train_losses_base=train_losses_base,
                    train_losses_reg=train_losses_reg,
                    logger_info=logger_info,
                    wandb_info=wandb_info,
                )
            )

            if self._check_early_stopping():
                break

        return train_losses, valid_losses, train_losses_base, train_losses_reg

    def _run_epoch(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: list[float],
        valid_losses: list[float],
        train_losses_base: list[float],
        train_losses_reg: list[float],
        logger_info: str = "",
        wandb_info: dict | None = None,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        if wandb_info is None:
            wandb_info = {}
        self.epoch_counter += 1

        # Tell DistributedSampler which epoch we're in so each rank
        # sees a different shuffle order per epoch.
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(self.epoch_counter)

        # Periodic reactivation recalibration: recompute (m, b) from the
        # current voltage distribution every N epochs using the same
        # quantile formula as the init calibration. This keeps the gate
        # matched to the evolving V distribution and eliminates drift.
        recalib_every = getattr(self, "recalibrate_reactivation_every", 0)
        recalib_start = max(
            1, int(getattr(self, "reactivation_recalibration_start_epoch", 1))
        )
        if (
            recalib_every > 0
            and self.epoch_counter >= recalib_start
            and (self.epoch_counter - recalib_start) % recalib_every == 0
        ):
            self._recalibrate_reactivation(model, train_loader)

        train_loss_total, train_loss_base, train_loss_reg = self._epoch_train(
            model, train_loader
        )
        train_losses.append(train_loss_total)
        train_losses_base.append(train_loss_base)
        train_losses_reg.append(train_loss_reg)
        # Under DDP, all ranks validate on disjoint shards with equal-length
        # padded samplers and then all-reduce exact sample-weighted loss.
        if self._is_distributed:
            unwrapped = self._unwrap_model(model)
            valid_loss = self._epoch_valid_distributed(unwrapped, valid_loader)
        else:
            valid_loss = self._non_distributed_validation_loss(model, valid_loader)

        valid_losses.append(valid_loss)

        self._update_best(model, valid_loss)
        self._save_checkpoint(model)
        self._analyze()
        self._tracking(
            train_loss_total,
            valid_loss,
            logger_info,
            wandb_info,
            train_loss_base=train_loss_base,
            train_loss_reg=train_loss_reg,
        )
        return train_losses, valid_losses, train_losses_base, train_losses_reg

    def _recalibrate_reactivation(
        self, model: BaseModel, train_loader: DataLoader
    ) -> None:
        """Recompute (m, b) from the current voltage distribution.

        Uses the same calibration function as the init-time calibration
        but runs it on live training data with the current weights. This
        keeps the gate matched to the evolving V distribution.
        """
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
            calibrate_reactivation_from_data,
        )

        mode = self._resolve_reactivation_recalibration_mode(model)
        n_batches = max(
            1, int(getattr(self, "reactivation_recalibration_num_batches", 3))
        )
        ema_alpha = float(getattr(self, "reactivation_recalibration_ema_alpha", 1.0))

        # Collect a few batches from the current train loader
        batches = []
        for i, batch in enumerate(train_loader):
            if i >= n_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            batches.append(x.to(self.device))

        if not batches:
            return

        diag = calibrate_reactivation_from_data(
            model,
            batches,
            k=None,
            device=self.device,
            mode=mode,
            ema_alpha=ema_alpha,
        )

        if self._is_main_process and diag:
            m_values = [d["m"] for d in diag.values()]
            applied_m_values = [d.get("m_applied", d["m"]) for d in diag.values()]
            logger.info(
                f"{self.logger_info_prefix}Recalibrated reactivation at epoch "
                f"{self.epoch_counter}: {len(diag)} layers, "
                f"m target range [{min(m_values):.3f}, {max(m_values):.3f}], "
                f"applied m range [{min(applied_m_values):.3f}, {max(applied_m_values):.3f}], "
                f"batches={n_batches}, ema_alpha={max(0.0, min(1.0, ema_alpha)):.3f}"
            )

    def _epoch_train(
        self, model: BaseModel, train_loader: DataLoader
    ) -> tuple[float, float, float]:
        model.train()
        train_loss_total = 0
        train_loss_base = 0
        train_loss_reg = 0
        n_batches = 0

        # Check once if regularization is enabled to avoid repeated checks
        has_regularization = self.regularization_manager.has_regularization()
        profile_loader = bool(getattr(self, "profile_dataloader", False))
        data_wait_total = 0.0
        step_time_total = 0.0
        max_data_wait = 0.0

        train_iter = iter(train_loader)
        while True:
            fetch_start = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            if profile_loader:
                data_wait = time.perf_counter() - fetch_start
                data_wait_total += data_wait
                max_data_wait = max(max_data_wait, data_wait)
                step_start = time.perf_counter()

            x_batch, y_batch = self._move_training_batch_to_device(
                batch,
                train_loader,
            )

            if self.enable_amp:
                # Mixed precision forward pass
                with autocast():
                    base_loss = self.loss_function(model, x_batch, y_batch)

                    # Add all regularization losses only if regularization is enabled
                    if has_regularization:
                        reg_loss = (
                            self.regularization_manager.compute_regularization_loss(
                                model,
                                x_batch,
                                y_batch,
                                param_groups=self.optimizer.param_groups,
                            )
                        )
                        total_loss = base_loss + reg_loss
                        reg_loss_item = reg_loss.item()
                    else:
                        total_loss = base_loss
                        reg_loss_item = 0.0

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()

                # Optional expensive operations (configurable for performance)
                if self.enable_nan_checking:
                    nan_detected = self._check_and_handle_nan_gradients(model)
                    if nan_detected:
                        logger.warning(
                            "NaNs detected and handled in gradients during AMP training"
                        )

                # Gradient clipping with AMP
                if self.grad_clip_value:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    if self.enable_adaptive_clipping:
                        self._apply_adaptive_gradient_clipping(model)
                    else:
                        clip_grad_value_(model.parameters(), self.grad_clip_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                base_loss: torch.Tensor = self.loss_function(model, x_batch, y_batch)

                # Add all regularization losses only if regularization is enabled
                if has_regularization:
                    reg_loss = self.regularization_manager.compute_regularization_loss(
                        model,
                        x_batch,
                        y_batch,
                        param_groups=self.optimizer.param_groups,
                    )
                    total_loss = base_loss + reg_loss
                    reg_loss_item = reg_loss.item()
                else:
                    total_loss = base_loss
                    reg_loss_item = 0.0

                self.optimizer.zero_grad()
                total_loss.backward()

                # Optional expensive operations (configurable for performance)
                if self.enable_nan_checking:
                    nan_detected = self._check_and_handle_nan_gradients(model)
                    if nan_detected:
                        logger.warning("NaNs detected and handled in gradients")

                # Gradient clipping: adaptive (expensive) vs simple (fast)
                if self.enable_adaptive_clipping:
                    self._apply_adaptive_gradient_clipping(model)
                else:
                    # Use simple, fast gradient clipping (default)
                    clip_grad_value_(model.parameters(), self.grad_clip_value)
                self.optimizer.step()

            train_loss_total += total_loss.item()
            train_loss_base += base_loss.item()
            train_loss_reg += reg_loss_item
            n_batches += 1
            if profile_loader:
                step_time_total += time.perf_counter() - step_start

        local_n_batches = n_batches
        if self._is_distributed:
            stats = torch.tensor(
                [
                    float(train_loss_total),
                    float(train_loss_base),
                    float(train_loss_reg),
                    float(n_batches),
                ],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            train_loss_total, train_loss_base, train_loss_reg, n_batches = (
                stats[0].item(),
                stats[1].item(),
                stats[2].item(),
                int(stats[3].item()),
            )

        if profile_loader and self._is_main_process:
            self._log_dataloader_profile(
                n_batches=local_n_batches,
                data_wait_total=data_wait_total,
                step_time_total=step_time_total,
                max_data_wait=max_data_wait,
            )

        denom = max(n_batches, 1)
        train_loss_total /= denom
        train_loss_base /= denom
        train_loss_reg /= denom
        return train_loss_total, train_loss_base, train_loss_reg

    def _loader_uses_non_blocking_transfer(self, loader: DataLoader) -> bool:
        """Return whether batch copies can be issued asynchronously."""
        try:
            device_type = torch.device(self.device).type
        except (RuntimeError, TypeError):
            return False
        return device_type == "cuda" and bool(getattr(loader, "pin_memory", False))

    def _move_training_batch_to_device(
        self,
        batch,
        loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move the input/target tensors using the loader's transfer policy."""
        non_blocking = self._loader_uses_non_blocking_transfer(loader)
        return (
            batch[0].to(self.device, non_blocking=non_blocking),
            batch[1].to(self.device, non_blocking=non_blocking),
        )

    def _log_dataloader_profile(
        self,
        *,
        n_batches: int,
        data_wait_total: float,
        step_time_total: float,
        max_data_wait: float,
    ) -> None:
        """Log epoch-level dataloader wait time for input-pipeline diagnosis."""
        if n_batches <= 0:
            return
        total_time = data_wait_total + step_time_total
        wait_fraction = data_wait_total / max(total_time, 1e-12)
        logger.info(
            "%sDataLoader profile epoch=%s batches=%s "
            "data_wait=%.3fs step_time=%.3fs wait_fraction=%.1f%% "
            "avg_wait=%.1fms max_wait=%.1fms",
            getattr(self, "logger_info_prefix", "[Standard] "),
            getattr(self, "epoch_counter", 0),
            n_batches,
            data_wait_total,
            step_time_total,
            100.0 * wait_fraction,
            1000.0 * data_wait_total / n_batches,
            1000.0 * max_data_wait,
        )

    def _epoch_valid_stats(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> _ValidationLossStats:
        """Collect explicit batch- and sample-level validation aggregates."""
        model.eval()
        batch_loss_sum = 0.0
        n_batches = 0
        sample_weighted_loss_sum = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch in valid_loader:
                x_batch, y_batch = self._move_training_batch_to_device(
                    batch,
                    valid_loader,
                )
                loss: torch.Tensor = self.loss_function(model, x_batch, y_batch)
                loss_value = loss.item()
                batch_size = y_batch.size(0)
                batch_loss_sum += loss_value
                n_batches += 1
                sample_weighted_loss_sum += loss_value * batch_size
                n_samples += batch_size
        return _ValidationLossStats(
            batch_loss_sum=batch_loss_sum,
            batch_count=n_batches,
            sample_weighted_loss_sum=sample_weighted_loss_sum,
            sample_count=n_samples,
        )

    def _epoch_valid(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> tuple[float, int]:
        """Return the legacy ``(batch_loss_sum, batch_count)`` tuple."""
        stats = self._epoch_valid_stats(model, valid_loader)
        return stats.batch_loss_sum, stats.batch_count

    def _validation_loss_reduction(self) -> str | None:
        """Return the configured scalar-loss reduction, when declared."""
        reduction = getattr(self.loss_function, "reduction", None)
        return str(reduction) if reduction is not None else None

    def _non_distributed_validation_loss(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> float:
        """Return the checkpoint metric for non-distributed validation.

        Mean-reduced losses are weighted by the number of examples in each
        batch, matching evaluation of the complete dataset when the final
        batch is smaller. Other or undeclared reductions retain the historical
        mean-over-batches behavior.
        """
        stats = self._epoch_valid_stats(model, valid_loader)
        if self._validation_loss_reduction() == "mean":
            return stats.sample_mean
        return stats.batch_mean

    def _epoch_valid_distributed(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> float:
        """Return exact global validation loss under DDP.

        All ranks process equal-length validation shards via
        ``PaddedDistributedEvalSampler``. Padding entries are excluded from the
        reduced sample-weighted loss.
        """
        model.eval()
        weighted_loss_sum = 0.0
        total_examples = 0
        sampler = valid_loader.sampler
        if isinstance(sampler, PaddedDistributedEvalSampler):
            real_count = len(sampler.real_indices)
        else:
            real_count = None

        seen = 0
        with torch.no_grad():
            for batch in valid_loader:
                x_batch, y_batch = self._move_training_batch_to_device(
                    batch,
                    valid_loader,
                )
                batch_size = y_batch.size(0)
                if real_count is not None:
                    real_in_batch = min(batch_size, real_count - seen)
                    if real_in_batch <= 0:
                        seen += batch_size
                        continue
                    if real_in_batch < batch_size:
                        x_batch = x_batch[:real_in_batch]
                        y_batch = y_batch[:real_in_batch]
                        batch_size = real_in_batch
                    seen += batch_size
                loss: torch.Tensor = self.loss_function(model, x_batch, y_batch)
                weighted_loss_sum += loss.item() * batch_size
                total_examples += batch_size
        stats = torch.tensor(
            [weighted_loss_sum, float(total_examples)],
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats[0].item() / max(stats[1].item(), 1.0)

    def _evaluate_loader_loss(self, model: BaseModel, loader: DataLoader) -> float:
        if self._is_distributed:
            return self._epoch_valid_distributed(self._unwrap_model(model), loader)
        return self._non_distributed_validation_loss(model, loader)

    def _apply_pruning_and_evaluate(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict:
        """Apply configured pruning after training and measure pre/post loss."""
        if not hasattr(self, "device"):
            self._initialize_attributes(model, self.epochs)

        train_loader, valid_loader = self._dataloaders(train_data, valid_data)
        unwrapped = self._unwrap_model(model)

        pre_train_loss = self._evaluate_loader_loss(model, train_loader)
        pre_valid_loss = self._evaluate_loader_loss(model, valid_loader)
        pruning_stats = self.pruning_manager.prune_model(unwrapped)
        post_train_loss = self._evaluate_loader_loss(model, train_loader)
        post_valid_loss = self._evaluate_loader_loss(model, valid_loader)

        return {
            "pruning_stats": pruning_stats,
            "pre_pruning_performance": {
                "train_loss": pre_train_loss,
                "valid_loss": pre_valid_loss,
            },
            "post_pruning_performance": {
                "train_loss": post_train_loss,
                "valid_loss": post_valid_loss,
            },
        }

    def _update_best(self, model: BaseModel, valid_loss: float) -> None:
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_epoch = self.epoch_counter
            self.best_state_dict = deepcopy(self._unwrap_model(model).state_dict())
            self.patience_counter = 0
            if (
                self._is_main_process
                and self.save_path
                and os.path.exists(self.save_path)
            ):
                torch.save(
                    self.best_state_dict,
                    os.path.join(
                        self.save_path, f"{self.filename_prefix}best_model.pt"
                    ),
                )
        else:
            self.patience_counter += 1

    def _save_checkpoint(self, model: BaseModel):
        if not self._is_main_process:
            return
        if self.checkpointing and (self.epoch_counter % self.checkpoint_interval == 0):
            checkpoint_path = os.path.join(
                self.save_path, f"{self.filename_prefix}checkpoints"
            )
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(
                self._unwrap_model(model).state_dict(),
                os.path.join(checkpoint_path, f"epoch_{self.epoch_counter}.pt"),
            )

    def _analyze(self):
        if not self._is_main_process:
            return
        if self.analysis_manager is not None:
            # Lazy import to avoid circular dependency
            from dendritic_modeling.analysis import AnalysisManager

            if isinstance(self.analysis_manager, AnalysisManager):
                self.analysis_manager.run_analysis(
                    filename=f"epoch{self.epoch_counter}", training=True
                )

    def _tracking(
        self,
        train_loss: float,
        valid_loss: float,
        logger_info: str = "",
        wandb_info: dict | None = None,
        train_loss_base: Optional[float] = None,
        train_loss_reg: Optional[float] = None,
    ) -> None:
        if wandb_info is None:
            wandb_info = {}
        elapsed = time.time() - self.start_time
        if not self.suppress_prints and (self.epoch_counter % self.print_every == 0):
            if train_loss_base is not None and train_loss_reg is not None:
                logger_info = (
                    logger_info
                    + f"epoch={self.epoch_counter}/{self.total_epochs}, "
                    + f"train_loss={train_loss:.4f} (base={train_loss_base:.4f}, reg={train_loss_reg:.4f}), "
                    + f"valid_loss={valid_loss:.4f}, "
                    + f"elapsed={elapsed:.1f}s"
                )
            else:
                logger_info = (
                    logger_info
                    + f"epoch={self.epoch_counter}/{self.total_epochs}, "
                    + f"train_loss={train_loss:.4f}, "
                    + f"valid_loss={valid_loss:.4f}, "
                    + f"elapsed={elapsed:.1f}s"
                )
            logger.info(logger_info)

    def _check_early_stopping(self) -> bool:
        if self.early_stopping and self.patience_counter >= self.patience:
            if not self.suppress_prints:
                logger.info(
                    f"Early stopping triggered after {self.epoch_counter} epochs."
                )
            return True
        else:
            return False

    def _loss_plotting(
        self,
        train_losses: list[float],
        valid_losses: list[float],
        train_losses_base: Optional[list[float]] = None,
        train_losses_reg: Optional[list[float]] = None,
    ) -> None:
        if not self._is_main_process:
            return
        if self.plot_losses and len(train_losses) > 1:
            if self.save_path and os.path.exists(self.save_path):
                plot_loss_curves(
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    loss_name=self.loss_function._loss_name,
                    save_dir=self.save_path,
                    filename_prefix=self.filename_prefix,
                    train_losses_base=train_losses_base,
                    train_losses_reg=train_losses_reg,
                )

    def _check_and_handle_nan_gradients(self, model: BaseModel) -> bool:
        """
        Check for NaN values in gradients and handle them to prevent training crashes.


        Args:
            model: The model to check gradients for

        Returns:
            bool: True if NaNs were detected and handled, False otherwise
        """
        nan_detected = False

        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_detected = True
                    # Replace NaNs with zeros to prevent propagation
                    param.grad = torch.nan_to_num(
                        param.grad, nan=0.0, posinf=1.0, neginf=-1.0
                    )

                    logger.warning(f"NaNs detected in gradients for parameter: {name}")

        return nan_detected

    def _apply_adaptive_gradient_clipping(self, model: BaseModel) -> float:
        """
        Apply adaptive gradient clipping with dynamic learning rate adjustment.


        Args:
            model: The model to apply gradient clipping to

        Returns:
            float: The gradient norm before clipping
        """
        if not self.grad_clip_value:
            return 0.0

        # Calculate gradient norm for adaptive clipping (iterates all parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.grad_clip_value
        )
        return grad_norm


__all__ = ["Trainer"]
