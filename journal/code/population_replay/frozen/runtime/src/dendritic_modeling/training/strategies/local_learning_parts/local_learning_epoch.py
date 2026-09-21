"""Epoch-level helpers for local learning training."""

from functools import partial
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_gradient_utils import (
    _accumulate_parameter_grad,
)
from dendritic_modeling.utils.hooks import (
    iter_modules_of_type,
    remove_hook_handles,
    run_with_forward_hooks,
)


def _has_dendritic_branch_layers(model: BaseModel) -> bool:
    try:
        return any(iter_modules_of_type(model, DendriticBranchLayer))
    except Exception:
        return False


def _supports_local_learning_train(model: BaseModel) -> bool:
    has_ei = hasattr(model, "core_network") and isinstance(
        model.core_network, ExcitationInhibitionNetwork
    )
    return has_ei or _has_dendritic_branch_layers(model)


def _resolve_freeze_update_modes(
    local_cfg: Any,
    epoch_counter: int,
) -> tuple[bool, str, str]:
    """Resolve update modes after applying inclusive freeze windows."""

    update_reactivation = local_cfg.update_reactivation
    encoder_update_mode = local_cfg.encoder_update_mode
    decoder_update_mode = local_cfg.decoder_update_mode

    if epoch_counter <= max(0, int(local_cfg.freeze_encoder_epochs)):
        encoder_update_mode = "none"
    if epoch_counter <= max(0, int(local_cfg.freeze_reactivation_epochs)):
        update_reactivation = False
    if epoch_counter <= max(0, int(local_cfg.freeze_decoder_epochs)):
        decoder_update_mode = "none"

    return update_reactivation, encoder_update_mode, decoder_update_mode


def _resolve_schedule_epoch_range(schedule_entry: Any) -> tuple[int, int]:
    """Resolve a schedule entry's inclusive epoch range with legacy fallback."""

    try:
        start_epoch = int(schedule_entry.get("start_epoch", 1))
        end_epoch = int(schedule_entry.get("end_epoch", 10**9))
    except Exception:
        start_epoch, end_epoch = 1, 10**9
    return start_epoch, end_epoch


def _apply_training_schedule_update_modes(
    *,
    training_schedule: list[Any],
    epoch_counter: int,
    update_reactivation: bool,
    encoder_update_mode: str,
    decoder_update_mode: str,
) -> tuple[bool, str, str]:
    """Apply the first matching explicit local-learning training schedule entry."""

    for sched in training_schedule:
        start_epoch, end_epoch = _resolve_schedule_epoch_range(sched)
        if start_epoch <= epoch_counter <= end_epoch:
            if "update_reactivation" in sched:
                update_reactivation = bool(sched["update_reactivation"])
            if "encoder_update_mode" in sched:
                encoder_update_mode = str(sched["encoder_update_mode"]).lower()
            if "decoder_update_mode" in sched:
                decoder_update_mode = str(sched["decoder_update_mode"]).lower()
            break

    return update_reactivation, encoder_update_mode, decoder_update_mode


def _resolve_epoch_update_modes(
    local_cfg,
    epoch_counter: int,
) -> tuple[bool, str, str]:
    update_reactivation, encoder_update_mode, decoder_update_mode = (
        _resolve_freeze_update_modes(local_cfg, epoch_counter)
    )

    if local_cfg.training_schedule:
        update_reactivation, encoder_update_mode, decoder_update_mode = (
            _apply_training_schedule_update_modes(
                training_schedule=local_cfg.training_schedule,
                epoch_counter=epoch_counter,
                update_reactivation=update_reactivation,
                encoder_update_mode=encoder_update_mode,
                decoder_update_mode=decoder_update_mode,
            )
        )

    return update_reactivation, encoder_update_mode, decoder_update_mode


def _resolve_reactivation_update_enabled(
    update_reactivation: bool,
    reactivation_update_mode: str,
) -> bool:
    """Gate local reactivation updates by the shared trainer update mode."""

    if str(reactivation_update_mode).lower() == "backprop":
        return bool(update_reactivation)
    return False


def _resolve_local_epoch_update_modes(
    local_cfg: Any,
    epoch_counter: int,
    reactivation_update_mode: str,
) -> tuple[bool, str, str]:
    """Resolve epoch update modes after applying the local reactivation gate."""

    update_reactivation, encoder_update_mode, decoder_update_mode = (
        _resolve_epoch_update_modes(local_cfg, epoch_counter)
    )
    update_reactivation = _resolve_reactivation_update_enabled(
        update_reactivation,
        reactivation_update_mode,
    )
    return update_reactivation, encoder_update_mode, decoder_update_mode


def _move_batch_to_device(
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move the local-learning input and target tensors to the trainer device."""

    return x_batch.to(device), y_batch.to(device)


def _accumulate_local_logging_loss(
    *,
    train_loss: float,
    train_loss_base: float,
    train_loss_reg: float,
    loss: torch.Tensor,
) -> tuple[float, float, float]:
    """Accumulate local-learning logging loss totals for one batch."""

    loss_item = loss.item()
    return train_loss + loss_item, train_loss_base + loss_item, train_loss_reg


def _average_local_epoch_losses(
    *,
    train_loss: float,
    train_loss_base: float,
    train_loss_reg: float,
    n_batches: int,
    device: torch.device | str | None = None,
) -> tuple[float, float, float]:
    """Average local-learning epoch loss totals over batches and DDP ranks."""

    if (
        device is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        totals = torch.tensor(
            [
                train_loss,
                train_loss_base,
                train_loss_reg,
                float(n_batches),
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        train_loss = float(totals[0].item())
        train_loss_base = float(totals[1].item())
        train_loss_reg = float(totals[2].item())
        n_batches = int(totals[3].item())

    return (
        train_loss / n_batches,
        train_loss_base / n_batches,
        train_loss_reg / n_batches,
    )


def _collect_topk_modules(model: BaseModel) -> list[TopKLinear]:
    return list(iter_modules_of_type(model, TopKLinear))


def _set_topk_mask_cache(
    topk_modules: list[TopKLinear],
    enabled: bool,
) -> None:
    for module in topk_modules:
        module.cache_mask = enabled


def _enable_topk_mask_cache(topk_modules: list[TopKLinear]) -> None:
    _set_topk_mask_cache(topk_modules, True)


def _clear_topk_mask_cache(topk_modules: list[TopKLinear]) -> None:
    _set_topk_mask_cache(topk_modules, False)
    for module in topk_modules:
        module._last_forward_weight_mask = None


def _remove_local_recorders(
    trainer: Any,
    handles: list[torch.utils.hooks.RemovableHandle],
) -> None:
    remove_forward_hooks = getattr(trainer, "remove_forward_hooks", None)
    if callable(remove_forward_hooks):
        remove_forward_hooks(handles)
    else:
        remove_hook_handles(handles)


def _is_class_index_target(y_batch: torch.Tensor) -> bool:
    """Return whether a target tensor stores class indices rather than soft labels."""

    return y_batch.dim() == 1 or (y_batch.dim() == 2 and y_batch.size(1) == 1)


def _prepare_label_hsic_target(
    y_batch: torch.Tensor,
    y_hat: torch.Tensor,
) -> torch.Tensor:
    """Prepare label-derived HSIC targets with legacy one-hot handling."""

    if _is_class_index_target(y_batch):
        num_classes = y_hat.size(-1)
        return torch.nn.functional.one_hot(
            y_batch.view(-1),
            num_classes=num_classes,
        ).to(y_hat.dtype)
    return y_batch.to(y_hat.dtype)


def _prepare_hsic_target(
    local_cfg,
    y_batch: torch.Tensor,
    y_hat: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not local_cfg.hsic.enabled:
        return None

    if local_cfg.hsic.target_source == "labels":
        return _prepare_label_hsic_target(y_batch, y_hat)

    return y_hat.detach()


def _zero_optimizer_gradients(optimizer: torch.optim.Optimizer) -> None:
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def _collect_trainable_parameters(module: nn.Module) -> list[torch.nn.Parameter]:
    """Return trainable parameters in module order."""

    return [param for param in module.parameters() if param.requires_grad]


def _collect_backprop_param_groups(
    model: BaseModel,
    encoder_update_mode: str,
    decoder_update_mode: str,
) -> list[list[torch.nn.Parameter]]:
    component_model = model.module if hasattr(model, "module") else model
    backprop_param_groups: list[list[torch.nn.Parameter]] = []
    if (
        encoder_update_mode == "backprop"
        and hasattr(component_model, "encoder_network")
        and component_model.encoder_network is not None
    ):
        enc_params = _collect_trainable_parameters(component_model.encoder_network)
        if enc_params:
            backprop_param_groups.append(enc_params)

    if (
        hasattr(component_model, "decoder_network")
        and decoder_update_mode == "backprop"
    ):
        dec_params = _collect_trainable_parameters(component_model.decoder_network)
        if dec_params:
            backprop_param_groups.append(dec_params)

    return backprop_param_groups


def _accumulate_detached_parameter_grad(
    param: torch.nn.Parameter,
    grad_value: torch.Tensor | None,
) -> None:
    """Accumulate a detached autograd gradient with legacy clone-on-first semantics."""

    if grad_value is None:
        return
    grad = grad_value.detach()
    if param.grad is None:
        param.grad = grad.clone()
    else:
        param.grad = param.grad + grad


def _accumulate_backprop_gradients(
    loss: torch.Tensor,
    backprop_param_groups: list[list[torch.nn.Parameter]],
) -> None:
    for group_idx, params in enumerate(backprop_param_groups):
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=group_idx < len(backprop_param_groups) - 1,
            allow_unused=True,
        )
        for param, grad_value in zip(params, grads):
            _accumulate_detached_parameter_grad(param, grad_value)


def _accumulate_backprop_update_gradients(
    *,
    loss: torch.Tensor,
    model: BaseModel,
    encoder_update_mode: str,
    decoder_update_mode: str,
) -> None:
    """Accumulate autograd gradients requested by encoder/decoder update modes."""

    backprop_param_groups = _collect_backprop_param_groups(
        model=model,
        encoder_update_mode=encoder_update_mode,
        decoder_update_mode=decoder_update_mode,
    )
    _accumulate_backprop_gradients(loss, backprop_param_groups)


def _decoder_local_scale(
    normalize_by_batch: bool,
    batch_size: int,
) -> float:
    """Return decoder-local batch scaling with legacy empty-batch behavior."""

    if normalize_by_batch and batch_size > 0:
        return 1.0 / float(batch_size)
    return 1.0


def _compute_decoder_local_gradients(
    *,
    h_in: torch.Tensor,
    delta_out: torch.Tensor,
    normalize_by_batch: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local decoder weight and bias gradients from cached activations."""

    e_out = delta_out.detach()  # [B, out]
    scale = _decoder_local_scale(normalize_by_batch, e_out.size(0))
    grad_w = scale * (e_out.t() @ h_in.detach())  # [out, in]
    grad_b = scale * e_out.sum(dim=0)
    return grad_w, grad_b


def _apply_decoder_local_gradients(
    lin_mod: Any,
    h_in: Any,
    delta_out: torch.Tensor,
    normalize_by_batch: bool,
) -> None:
    if not isinstance(lin_mod, nn.Linear) or not isinstance(h_in, torch.Tensor):
        return

    grad_w, grad_b = _compute_decoder_local_gradients(
        h_in=h_in,
        delta_out=delta_out,
        normalize_by_batch=normalize_by_batch,
    )
    _accumulate_parameter_grad(lin_mod.weight, grad_w)

    if lin_mod.bias is not None:
        _accumulate_parameter_grad(lin_mod.bias, grad_b)


def _apply_local_decoder_update(
    *,
    model: BaseModel,
    decoder_update_mode: str,
    decoder_cache: dict[str, Any],
    delta_out: torch.Tensor,
    normalize_by_batch: bool,
) -> None:
    """Apply decoder-local gradients when the current epoch mode requests them."""

    component_model = model.module if hasattr(model, "module") else model
    if not (
        hasattr(component_model, "decoder_network") and decoder_update_mode == "local"
    ):
        return

    _apply_decoder_local_gradients(
        lin_mod=decoder_cache.get("module"),
        h_in=decoder_cache.get("input"),
        delta_out=delta_out,
        normalize_by_batch=normalize_by_batch,
    )


def _should_clip_gradients(clip_grad_value: Any) -> bool:
    """Return whether value-wise gradient clipping is enabled."""

    return bool(clip_grad_value and clip_grad_value > 0)


def _clip_model_gradients(model: BaseModel, clip_grad_value: Any) -> None:
    if _should_clip_gradients(clip_grad_value):
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)


def _average_distributed_gradients(model: BaseModel) -> None:
    """Average manually assigned local gradients across distributed ranks."""

    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size <= 1:
        return

    parameters = list(model.parameters())
    if not parameters:
        return
    device = parameters[0].device
    present = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(present, op=dist.ReduceOp.SUM)
    inconsistent = (present != 0) & (present != world_size)
    if bool(inconsistent.any()):
        indices = inconsistent.nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(
            "Local-gradient presence differs across distributed ranks for "
            f"parameter indices {indices}"
        )

    for parameter, rank_count in zip(parameters, present.tolist()):
        if rank_count == 0:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(world_size))


def _forward_with_local_recorders(
    trainer: Any,
    model: BaseModel,
    x_batch: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    layer_records, handles = trainer._attach_local_recorders(model)
    topk_modules = _collect_topk_modules(model)
    _enable_topk_mask_cache(topk_modules)

    def _cleanup_recorders(
        active_handles: list[torch.utils.hooks.RemovableHandle],
    ) -> None:
        _remove_local_recorders(trainer, active_handles)
        _clear_topk_mask_cache(topk_modules)

    y_hat: torch.Tensor = run_with_forward_hooks(
        attach=lambda: handles,
        remove=_cleanup_recorders,
        body=partial(model, x_batch),
    )

    return y_hat, layer_records


def _resolve_stdp_error_signals(
    local_cfg: Any,
    e_n: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    stdp_error_signal = e_n
    if str(getattr(local_cfg, "rule_variant", "")).lower() == "stdp":
        e_n = torch.zeros_like(e_n)
    return e_n, stdp_error_signal


def _apply_path_propagation_factor(
    local_cfg: Any,
    broadcast_state: Any,
    rec: dict[str, Any],
    e_n: torch.Tensor,
) -> torch.Tensor:
    if (
        local_cfg.morphology_aware.use_path_propagation
        and broadcast_state.mode != "path_transport"
    ):
        path_factor = rec.get("path_factor", 1.0)
        if isinstance(path_factor, torch.Tensor):
            return e_n * path_factor
    return e_n
