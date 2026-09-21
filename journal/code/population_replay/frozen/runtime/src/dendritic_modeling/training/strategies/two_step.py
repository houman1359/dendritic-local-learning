"""
Two-Step Training Strategies
============================

This module contains training strategies that implement two-step learning approaches,
where models are first pre-trained with local supervision and then fine-tuned end-to-end.
"""

import logging

import torch
import torch.nn.functional as functional
from torch.nn.utils import clip_grad_value_

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


def _initialize_local_heads(net_obj, model, device):
    """Initialize local supervision heads for each layer."""
    local_heads = []

    for i, layer in enumerate(net_obj.layers):
        # Determine output dimension of current layer
        out_dim = getattr(layer, "out_features", None)
        if out_dim is None:
            excit_sizes = getattr(net_obj, "excitatory_layer_sizes", None)
            if isinstance(excit_sizes, list) and i < len(excit_sizes):
                out_dim = excit_sizes[i]
            else:
                # Infer from dummy forward pass
                try:
                    dummy = torch.randn(1, getattr(net_obj, "input_dim", 784)).to(
                        device
                    )
                except Exception:
                    dummy = torch.randn(1, 784).to(device)
                dummy_out = layer(dummy)
                if isinstance(dummy_out, tuple):
                    dummy_out = dummy_out[0]
                out_dim = dummy_out.size(1)

        # Create local head
        if hasattr(layer, "weight"):
            device_for_head = layer.weight.device
        else:
            device_for_head = next(model.parameters()).device

        head = torch.nn.Linear(out_dim, getattr(model, "output_dim", 10)).to(
            device_for_head
        )
        local_heads.append(head)

    return local_heads


def _make_local_optimizers(net_obj, local_heads, base_optimizer):
    """Create per-layer optimizers that also own local-head parameters."""
    torch_optimizer = (
        base_optimizer.optimizer
        if hasattr(base_optimizer, "optimizer")
        and isinstance(base_optimizer.optimizer, torch.optim.Optimizer)
        else base_optimizer
    )
    optimizer_cls = type(torch_optimizer)
    defaults = dict(getattr(torch_optimizer, "defaults", {}))

    local_optimizers = []
    for layer, head in zip(net_obj.layers, local_heads):
        layer_params = list(layer.parameters()) if hasattr(layer, "parameters") else []
        params = [*layer_params, *list(head.parameters())]
        local_optimizers.append(optimizer_cls(params, **defaults))
    return local_optimizers


def _set_only_current_layer_and_head_trainable(model, layer, local_head):
    for param in model.parameters():
        param.requires_grad = False
    for param in local_head.parameters():
        param.requires_grad = True
    if hasattr(layer, "parameters"):
        for param in layer.parameters():
            param.requires_grad = True


def _current_local_params(layer, local_head):
    params = list(local_head.parameters())
    if hasattr(layer, "parameters"):
        params.extend(layer.parameters())
    return [param for param in params if param.requires_grad]


def _forward_through_local_layers(net_obj, x_batch, layer_idx):
    h_batch = x_batch
    for j in range(layer_idx + 1):
        layer = net_obj.layers[j]
        h_batch = layer(h_batch)
        if isinstance(h_batch, tuple):
            h_batch = h_batch[0]
        if hasattr(layer, "activation"):
            h_batch = layer.activation(h_batch)
    return h_batch


def _ensure_single_process_two_step(trainer_name: str, is_distributed: bool) -> None:
    if is_distributed:
        raise NotImplementedError(
            f"{trainer_name} staged local pretraining is not supported under DDP. "
            "Run the two-step pretraining phase on a single process or use "
            "strategy='standard' for distributed end-to-end training."
        )


def _resolve_layer_container(model):
    """Return the module that owns locally supervised layers."""
    for attr_name in ("core_network", "net", "network"):
        if hasattr(model, attr_name):
            return getattr(model, attr_name)
    return model


def _prepare_local_pretraining(trainer, model, net_obj, train_data, valid_data):
    train_loader, valid_loader = trainer._dataloaders(train_data, valid_data)
    trainer._initialize_attributes(model, trainer.pretrain_epochs + trainer.epochs)

    if not hasattr(trainer, "local_heads"):
        trainer.local_heads = _initialize_local_heads(net_obj, model, trainer.device)
    if not hasattr(trainer, "local_optimizers"):
        trainer.local_optimizers = _make_local_optimizers(
            net_obj, trainer.local_heads, trainer.optimizer
        )

    return train_loader, valid_loader


def _validate_full_model(trainer, model, valid_loader):
    with torch.no_grad():
        valid_loss = 0
        for batch in valid_loader:
            x_batch: torch.Tensor = batch[0].to(trainer.device)
            y_batch: torch.Tensor = batch[1].to(trainer.device)
            loss: torch.Tensor = trainer.loss_function(model, x_batch, y_batch)
            valid_loss += loss.item()

    return valid_loss / len(valid_loader)


def _restore_full_model_trainability(model):
    for param in model.parameters():
        param.requires_grad = True


def _prepend_pretrain_losses(final_results, train_losses, valid_losses):
    final_results["train_losses"] = train_losses + final_results["train_losses"]
    final_results["valid_losses"] = valid_losses + final_results["valid_losses"]
    return final_results


class TwoStepTrainer(Trainer):
    """
    Two-Step Training Strategy for dendritic networks.

    This trainer implements a two-phase training approach:
    1. Pre-training phase: Each layer is trained locally with local supervision heads
    2. Main training phase: End-to-end training of the full network

    This approach can help with gradient flow and learning in deep dendritic networks.
    """

    pretrain_epochs: int = 5

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train the model using two-step approach.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """
        _ensure_single_process_two_step(
            type(self).__name__,
            self._is_distributed,
        )

        net_obj = _resolve_layer_container(model)

        # Check if model has layers for local training
        if not hasattr(net_obj, "layers"):
            logger.info("No net.layers found => falling back to standard training.")
            return super().train(model, train_data, valid_data)

        # Training setup
        train_losses = []
        valid_losses = []

        train_loader, valid_loader = _prepare_local_pretraining(
            self, model, net_obj, train_data, valid_data
        )

        # Phase 1: Local pretraining
        logger.info(
            "=== TwoStepTrainer: Stage 1 "
            f"(local pretraining) for {self.pretrain_epochs} epochs ==="
        )
        n_layers = len(net_obj.layers)

        self.filename_prefix = "twostep_stage1_"
        try:
            for _ in range(1, self.pretrain_epochs + 1):
                self.epoch_counter += 1
                epoch_loss_accum = 0.0
                batch_counter = 0

                # Train each layer with its local head
                for layer_idx in range(n_layers):
                    layer = net_obj.layers[layer_idx]
                    local_head: torch.nn.Module = self.local_heads[layer_idx]
                    local_optimizer = self.local_optimizers[layer_idx]
                    _set_only_current_layer_and_head_trainable(model, layer, local_head)
                    local_params = _current_local_params(layer, local_head)

                    # Train on batches
                    for batch in train_loader:
                        x_batch: torch.Tensor = batch[0].to(self.device)
                        y_batch: torch.Tensor = batch[1].to(self.device)

                        # Forward pass through layers up to current layer
                        h_batch = _forward_through_local_layers(
                            net_obj, x_batch, layer_idx
                        )

                        # Local supervision
                        loss: torch.Tensor = self.loss_function(
                            local_head, h_batch, y_batch
                        )
                        local_optimizer.zero_grad()
                        loss.backward()
                        clip_grad_value_(local_params, self.grad_clip_value)
                        local_optimizer.step()

                        epoch_loss_accum += loss.item()
                        batch_counter += 1

                avg_loss = (
                    epoch_loss_accum / batch_counter if batch_counter > 0 else 0.0
                )
                train_losses.append(avg_loss)

                valid_loss = _validate_full_model(self, model, valid_loader)
                valid_losses.append(valid_loss)

                self._update_best(model, valid_loss)

                self._analyze()

                self._tracking(
                    avg_loss, valid_loss, "[TwoStep-Pretrain] ", {"phase": "pretrain"}
                )

                if self._check_early_stopping():
                    break
        except Exception:
            self.filename_prefix = ""
            _restore_full_model_trainability(model)
            raise

        self._loss_plotting(train_losses, valid_losses)

        # Phase 2: End-to-end training
        logger.info(
            "=== TwoStepTrainer: "
            "Stage 2 (end-to-end training) "
            f"for {self.epochs} epochs ==="
        )

        _restore_full_model_trainability(model)

        self.filename_prefix = "twostep_stage2_"
        try:
            final_results = super().train(model, train_data, valid_data)
        finally:
            self.filename_prefix = ""

        return _prepend_pretrain_losses(final_results, train_losses, valid_losses)


class TwoStepTrainerWithKL(Trainer):
    """
    Two-Step Training Strategy with KL Divergence Loss.

    Similar to TwoStepTrainer but uses a combined loss function during pre-training
    that includes both cross-entropy and KL divergence terms for better regularization.
    """

    pretrain_epochs: int = 5
    kl_weight: float = 0.1

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train the model using two-step approach with KL divergence.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """
        _ensure_single_process_two_step(
            type(self).__name__,
            self._is_distributed,
        )

        net_obj = _resolve_layer_container(model)

        # Check if model has layers for local training
        if not hasattr(net_obj, "layers"):
            logger.info("No net.layers found => falling back to standard training.")
            return super().train(model, train_data, valid_data)

        # Training setup
        train_losses = []
        valid_losses = []

        train_loader, valid_loader = _prepare_local_pretraining(
            self, model, net_obj, train_data, valid_data
        )

        # Phase 1: Local pretraining with KL divergence
        logger.info(
            "=== TwoStepTrainerWithKL: "
            "Stage 1 (local pretraining with KL) "
            f"for {self.pretrain_epochs} epochs ==="
        )
        n_layers = len(net_obj.layers)

        self.filename_prefix = "twostep_kl_stage1_"
        try:
            for _ in range(1, self.pretrain_epochs + 1):
                self.epoch_counter += 1
                epoch_loss_accum = 0.0
                batch_counter = 0

                # Train each layer with its local head
                for layer_idx in range(n_layers):
                    layer = net_obj.layers[layer_idx]
                    local_head: torch.nn.Module = self.local_heads[layer_idx]
                    local_optimizer = self.local_optimizers[layer_idx]
                    _set_only_current_layer_and_head_trainable(model, layer, local_head)
                    local_params = _current_local_params(layer, local_head)

                    # Train on batches
                    for batch in train_loader:
                        x_batch: torch.Tensor = batch[0].to(self.device)
                        y_batch: torch.Tensor = batch[1].to(self.device)

                        # Forward pass through layers up to current layer
                        h_batch = _forward_through_local_layers(
                            net_obj, x_batch, layer_idx
                        )

                        # Local head predictions
                        local_logits = local_head(h_batch)

                        # Get full model predictions for KL divergence
                        with torch.no_grad():
                            full_logits = model(x_batch)

                        # Combined loss: cross-entropy + KL divergence
                        ce_loss = functional.cross_entropy(local_logits, y_batch)
                        kl_loss = functional.kl_div(
                            functional.log_softmax(local_logits, dim=1),
                            functional.softmax(full_logits, dim=1),
                            reduction="batchmean",
                        )
                        loss = ce_loss + self.kl_weight * kl_loss

                        local_optimizer.zero_grad()
                        loss.backward()
                        clip_grad_value_(local_params, self.grad_clip_value)
                        local_optimizer.step()

                        epoch_loss_accum += loss.item()
                        batch_counter += 1

                avg_loss = (
                    epoch_loss_accum / batch_counter if batch_counter > 0 else 0.0
                )
                train_losses.append(avg_loss)

                valid_loss = _validate_full_model(self, model, valid_loader)
                valid_losses.append(valid_loss)

                self._update_best(model, valid_loss)

                self._analyze()

                self._tracking(
                    avg_loss,
                    valid_loss,
                    "[TwoStepKL-Pretrain] ",
                    {"phase": "pretrain_kl"},
                )

                if self._check_early_stopping():
                    break
        except Exception:
            self.filename_prefix = ""
            _restore_full_model_trainability(model)
            raise

        self._loss_plotting(train_losses, valid_losses)

        # Phase 2: End-to-end training (same as TwoStepTrainer)
        logger.info(
            "=== TwoStepTrainerWithKL: "
            "Stage 2 (end-to-end training) "
            f"for {self.epochs} epochs ==="
        )

        _restore_full_model_trainability(model)

        self.filename_prefix = "twostep_kl_stage2_"
        try:
            final_results = super().train(model, train_data, valid_data)
        finally:
            self.filename_prefix = ""

        return _prepend_pretrain_losses(final_results, train_losses, valid_losses)
