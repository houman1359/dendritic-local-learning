"""
Local learning training strategies.

This module implements multi-factor local learning rules for dendritic networks.

It provides biologically-inspired 3-/4-/5-factor Hebbian-like rules that use
locally available signals (pre-synaptic input, compartment voltages) modulated
by a broadcast soma error and optional morphology/information factors.
"""

import logging
import time
from typing import Any

import torch

from dendritic_modeling.config.local_learning import (
    build_local_rule_config,
    coerce_legacy_local_rule_config,
    section_to_dict,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_mixin import (
    LocalLearningBroadcastMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_epoch import (
    _accumulate_backprop_gradients as _accumulate_backprop_gradients,
    _accumulate_backprop_update_gradients,
    _accumulate_local_logging_loss,
    _apply_decoder_local_gradients as _apply_decoder_local_gradients,
    _apply_local_decoder_update,
    _apply_path_propagation_factor as _apply_path_propagation_factor,
    _average_distributed_gradients,
    _average_local_epoch_losses,
    _clear_topk_mask_cache as _clear_topk_mask_cache,
    _clip_model_gradients,
    _collect_backprop_param_groups as _collect_backprop_param_groups,
    _collect_topk_modules as _collect_topk_modules,
    _enable_topk_mask_cache as _enable_topk_mask_cache,
    _forward_with_local_recorders,
    _has_dendritic_branch_layers,
    _prepare_hsic_target,
    _resolve_epoch_update_modes as _resolve_epoch_update_modes,
    _resolve_local_epoch_update_modes,
    _resolve_reactivation_update_enabled,
    _resolve_stdp_error_signals as _resolve_stdp_error_signals,
    _supports_local_learning_train,
    _zero_optimizer_gradients,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_gradients_mixin import (
    LocalLearningGradientMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_hsic_mixin import (
    LocalLearningHSICMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_modulators import (
    LocalLearningModulatorMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_morphology_mixin import (
    LocalLearningMorphologyMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_recorders import (
    LocalLearningRecorderMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_rule_mixin import (
    LocalLearningRuleMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_signals import (
    LocalLearningSignalMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerStats,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_stdp_mixin import (
    LocalLearningSTDPMixin,
)
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


class LocalCreditAssignment(
    LocalLearningSignalMixin,
    LocalLearningRecorderMixin,
    LocalLearningSTDPMixin,
    LocalLearningBroadcastMixin,
    LocalLearningHSICMixin,
    LocalLearningRuleMixin,
    LocalLearningGradientMixin,
    LocalLearningMorphologyMixin,
    LocalLearningModulatorMixin,
    Trainer,
):
    """
    Local multi-factor learning strategy for dendritic networks.

    Implements 3-/4-/5-factor Hebbian-like local rules for both synaptic
    conductances (TopKLinear) and dendritic branch conductances (BlockLinear)
    using locally available signals plus a broadcast error from the soma.
    """

    def __init__(self, *args, **kwargs):
        config_dict = kwargs.pop("local_rule_config", {}) or {}
        super().__init__(*args, **kwargs)
        # Build LocalRuleConfig with proper nested config handling
        self.local_cfg = self._build_local_rule_config(config_dict)
        self._layer_stats: dict[int, _LayerStats] = {}
        self._decoder_cache: dict[str, Any] = {}
        self._additive_gain_cache: dict[str, Any] = {}
        self._additive_running_var: dict[int, torch.Tensor] = {}
        self._broadcast_cache: dict[tuple[Any, ...], Any] = {}
        self._stdp_traces: dict[int, dict[str, torch.Tensor]] = {}
        self._warned_decoder_soma_fallback = False
        self.filename_prefix = "local_learning_"
        self.logger_info_prefix = "[Local] "

    @staticmethod
    def _section_to_dict(section: Any) -> dict[str, Any]:
        """Convert a config section object to a plain dict."""
        return section_to_dict(section)

    def _coerce_legacy_local_rule_config(
        self, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize legacy flat local learning config to nested LocalRuleConfig layout."""
        return coerce_legacy_local_rule_config(config_dict)

    def _build_local_rule_config(self, config_dict: dict):
        """Build LocalRuleConfig from dict with proper nested config handling."""
        return build_local_rule_config(config_dict)

    def _should_update_local_record(self, rec: dict[str, Any]) -> bool:
        """Return whether LocalCA should write gradients for a recorded branch.

        This only changes explicit inhibitory-cell DendriNet populations. I-to-E
        synapses that live on excitatory dendrites remain ordinary excitatory
        population records and are still updated by LocalCA.
        """
        if rec.get("population") != "explicit_inhibitory":
            return True

        mode = getattr(self.local_cfg, "explicit_inhibitory_update_mode", None)
        if mode is None:
            # Compatibility for older LocalRuleConfig-like objects in tests.
            return bool(
                getattr(self.local_cfg, "update_explicit_inhibitory_cells", True)
            )
        return str(mode).lower() == "local_ca"

    # Override: do not run extra pretrain/final phases here
    def _pretrain(self, *args, **kwargs):  # type: ignore[override]
        return [], []

    def _resolve_local_reactivation_update_flag(
        self, update_reactivation: bool
    ) -> bool:
        """Disable local reactivation updates when the shared trainer mode does not use learned b,m."""
        return _resolve_reactivation_update_enabled(
            update_reactivation,
            self._normalize_reactivation_update_mode(),
        )

    # ----------------------- Core training overrides -----------------------
    def _epoch_train(
        self, model: BaseModel, train_loader
    ) -> tuple[float, float, float]:  # type: ignore[override]
        model.train()
        train_loss = 0.0
        train_loss_base = 0.0
        train_loss_reg = 0.0

        # Fallback: if model has no dendritic branch layers, use standard epoch_train
        if not _has_dendritic_branch_layers(model):
            # Defer to base implementation
            return super()._epoch_train(model, train_loader)

        profile_loader = bool(getattr(self, "profile_dataloader", False))
        data_wait_total = 0.0
        step_time_total = 0.0
        max_data_wait = 0.0
        n_batches = 0

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

            # Resolve per-epoch schedule for reactivation/decoder behavior
            update_reactivation, encoder_update_mode, decoder_update_mode = (
                _resolve_local_epoch_update_modes(
                    self.local_cfg,
                    self.epoch_counter,
                    self._normalize_reactivation_update_mode(),
                )
            )
            x_batch, y_batch = self._move_training_batch_to_device(
                batch,
                train_loader,
            )

            # Forward once to obtain outputs and local signals.
            y_hat, layer_records = _forward_with_local_recorders(self, model, x_batch)

            # Reuse the recorded forward for pointwise losses. Model-dependent
            # losses retain the compatibility fallback to a separate evaluation.
            loss = self._compute_logging_loss(
                model,
                x_batch,
                y_batch,
                predictions=y_hat,
            )
            train_loss, train_loss_base, train_loss_reg = (
                _accumulate_local_logging_loss(
                    train_loss=train_loss,
                    train_loss_base=train_loss_base,
                    train_loss_reg=train_loss_reg,
                    loss=loss,
                )
            )

            # Compute output-space error (dL/dy_hat) then resolve local soma-space
            # signals for local dendritic updates.
            delta_out: torch.Tensor = self._compute_soma_error(y_hat, y_batch)
            v0_local, delta_local = self._resolve_local_soma_signals(
                model=model,
                y_hat=y_hat,
                delta_out=delta_out.detach(),
            )

            # Prepare HSIC targets if needed
            y_target = _prepare_hsic_target(self.local_cfg, y_batch, y_hat)

            # Zero gradients (we will set param.grads directly)
            _zero_optimizer_gradients(self.optimizer)

            # Apply local learning updates as gradients on parameters
            self._apply_local_rule_gradients(
                model,
                layer_records,
                v0=v0_local,
                delta=delta_local,
                y_target=y_target,
                update_reactivation=update_reactivation,
            )

            _accumulate_backprop_update_gradients(
                loss=loss,
                model=model,
                encoder_update_mode=encoder_update_mode,
                decoder_update_mode=decoder_update_mode,
            )

            _apply_local_decoder_update(
                model=model,
                decoder_update_mode=decoder_update_mode,
                decoder_cache=self._decoder_cache,
                delta_out=delta_out,
                normalize_by_batch=self.local_cfg.normalize_by_batch,
            )

            # LocalCA assigns gradients directly rather than calling
            # ``loss.backward()``. DDP therefore cannot reduce them through
            # autograd hooks; average them explicitly before clipping.
            _average_distributed_gradients(model)

            # Optional gradient clipping (value-wise)
            _clip_model_gradients(model, self.local_cfg.clip_grad_value)

            # Step optimizer
            self.optimizer.step()
            n_batches += 1
            if profile_loader:
                step_time_total += time.perf_counter() - step_start

        if profile_loader and self._is_main_process:
            self._log_dataloader_profile(
                n_batches=n_batches,
                data_wait_total=data_wait_total,
                step_time_total=step_time_total,
                max_data_wait=max_data_wait,
            )

        return _average_local_epoch_losses(
            train_loss=train_loss,
            train_loss_base=train_loss_base,
            train_loss_reg=train_loss_reg,
            n_batches=n_batches,
            device=self.device,
        )

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        # Standard training loop but with local rule gradient computation
        if not _supports_local_learning_train(model):
            logger.info(
                "No EI/dendritic branch layers detected; falling back to standard training."
            )
            return super().train(model, train_data, valid_data)

        train_loader, valid_loader = self._dataloaders(train_data, valid_data)
        self._initialize_attributes(model, self.epochs)
        self._save_checkpoint(model)

        # Use the standard loop to ensure epoch counters and early stopping work
        train_losses: list[float] = []
        valid_losses: list[float] = []
        train_losses_base: list[float] = []
        train_losses_reg: list[float] = []
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

        if self.load_best_state_dict:
            model.load_state_dict(self.best_state_dict)

        # Log best model information like standard training
        logger.info(f"Best epoch: {self.best_epoch}, Best loss: {self.best_loss:.4f}")
        logger.info("Final model loaded from best checkpoint")

        results = {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_state_dict": self.best_state_dict,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
        }
        self._loss_plotting(train_losses, valid_losses)
        return results


__all__ = ["LocalCreditAssignment"]
