"""Soma-level Direct Feedback Alignment for dendritic networks.

This baseline projects output error through one fixed random matrix to the
somatic output of a dendritic core. The decoder receives its exact output-layer
gradient, while autograd differentiates the projected soma signal only within
the dendritic neuron. It is therefore a layer-level DFA baseline, not a claim
that credit is local inside the dendritic tree.
"""

from __future__ import annotations

import logging
import time
from functools import partial

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_epoch import (
    _average_local_epoch_losses,
)
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.utils.hooks import (
    iter_modules_of_type,
    remove_hook_handles,
    run_with_forward_hooks,
)

logger = logging.getLogger(__name__)


class SomaDFATrainer(Trainer):
    """Direct output-error feedback to somas with exact within-cell gradients."""

    def __init__(
        self,
        *args,
        feedback_seed: int | None = None,
        feedback_scale: float = 1.0,
        **kwargs,
    ):
        # A LocalCA config may remain present when a LocalCA experiment is
        # overridden from the command line. It has no role in this baseline.
        kwargs.pop("local_rule_config", None)
        super().__init__(*args, **kwargs)
        self.feedback_seed = self.seed if feedback_seed is None else int(feedback_seed)
        self.feedback_scale = float(feedback_scale)
        if self.feedback_scale <= 0:
            raise ValueError("feedback_scale must be positive")
        self._feedback_matrix: torch.Tensor | None = None
        self.filename_prefix = "soma_dfa_"
        self.logger_info_prefix = "[Soma-DFA] "

    @staticmethod
    def _first_decoder_linear(model: BaseModel) -> nn.Linear:
        component_model = Trainer._unwrap_model(model)
        decoder = getattr(component_model, "decoder_network", None)
        if decoder is None:
            raise ValueError("Soma-DFA requires a decoder network")
        first = next(iter_modules_of_type(decoder, nn.Linear), None)
        if first is None:
            raise ValueError("Soma-DFA requires at least one linear decoder layer")
        return first

    @staticmethod
    def _attach_soma_hook(
        decoder_linear: nn.Linear,
        store: dict[str, torch.Tensor],
    ) -> list[torch.utils.hooks.RemovableHandle]:
        def _capture(_module, inputs, _output):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("Could not capture the decoder's soma input")
            store["soma"] = inputs[0]

        return [decoder_linear.register_forward_hook(_capture)]

    def _feedback_for(
        self,
        *,
        soma_dim: int,
        output_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        expected_shape = (soma_dim, output_dim)
        cached = self._feedback_matrix
        if cached is not None:
            if tuple(cached.shape) != expected_shape:
                raise RuntimeError(
                    "Soma-DFA output geometry changed after feedback initialization: "
                    f"{tuple(cached.shape)} vs {expected_shape}"
                )
            return cached.to(device=device, dtype=dtype)

        generator = torch.Generator()
        generator.manual_seed(self.feedback_seed)
        feedback = torch.randn(expected_shape, generator=generator)
        feedback.mul_(self.feedback_scale / float(output_dim) ** 0.5)
        self._feedback_matrix = feedback.to(device=device, dtype=dtype)
        logger.info(
            "Initialized soma-DFA feedback matrix shape=%s seed=%s scale=%s",
            expected_shape,
            self.feedback_seed,
            self.feedback_scale,
        )
        return self._feedback_matrix

    def _dfa_surrogate(
        self,
        *,
        model: BaseModel,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        component_model = self._unwrap_model(model)
        decoder_linear = self._first_decoder_linear(component_model)
        soma_store: dict[str, torch.Tensor] = {}
        y_hat = run_with_forward_hooks(
            attach=partial(self._attach_soma_hook, decoder_linear, soma_store),
            remove=remove_hook_handles,
            body=partial(model, x_batch),
        )
        soma = soma_store.get("soma")
        if soma is None or soma.ndim != 2:
            raise RuntimeError("Soma-DFA requires a two-dimensional soma tensor")

        base_loss = self.loss_function.from_predictions(y_hat, y_batch)
        output_error = torch.autograd.grad(
            base_loss,
            y_hat,
            retain_graph=True,
            create_graph=False,
        )[0].detach()
        feedback = self._feedback_for(
            soma_dim=int(soma.size(1)),
            output_dim=int(y_hat.size(1)),
            device=soma.device,
            dtype=soma.dtype,
        )
        soma_error = output_error.to(dtype=soma.dtype) @ feedback.t()

        # Replay only the inexpensive decoder on detached somas. This supplies
        # its exact supervised gradient without allowing W^T error to enter the
        # dendritic core through the original decoder graph.
        decoder_logits = component_model.decoder_network(soma.detach())
        decoder_logits = (
            float(getattr(component_model, "fixed_output_scale", 1.0)) * decoder_logits
        )
        decoder_loss = self.loss_function.from_predictions(decoder_logits, y_batch)
        core_surrogate = (soma * soma_error).sum()
        return base_loss, decoder_loss + core_surrogate

    def _epoch_train(self, model: BaseModel, train_loader):
        if self.enable_amp:
            raise ValueError("Soma-DFA currently requires use_amp: false")
        if self.regularization_manager.has_regularization():
            raise ValueError("Soma-DFA does not yet support auxiliary regularization")

        component_model = self._unwrap_model(model)
        encoder = getattr(component_model, "encoder_network", None)
        if encoder is not None and any(p.requires_grad for p in encoder.parameters()):
            raise ValueError(
                "Soma-DFA currently requires a frozen or parameter-free encoder"
            )

        model.train()
        train_loss = 0.0
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

            x_batch, y_batch = self._move_training_batch_to_device(batch, train_loader)
            self.optimizer.zero_grad()
            base_loss, surrogate = self._dfa_surrogate(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
            )
            surrogate.backward()
            if self.grad_clip_value:
                clip_grad_value_(model.parameters(), self.grad_clip_value)
            self.optimizer.step()

            train_loss += float(base_loss.detach().item())
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
            train_loss_base=train_loss,
            train_loss_reg=0.0,
            n_batches=n_batches,
            device=self.device,
        )


__all__ = ["SomaDFATrainer"]
