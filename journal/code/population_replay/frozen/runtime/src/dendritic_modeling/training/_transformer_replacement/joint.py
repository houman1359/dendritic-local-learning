"""Joint language-model training losses and parameter policies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.config import Config
from dendritic_modeling.training._transformer_replacement.common import (
    resolve_configured_replacement_layers,
)
from dendritic_modeling.training.replacement_common import (
    _capture_requires_grad_states,
    _restore_requires_grad_states,
)


def _upper_tail_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Return empirical upper-tail CVaR over a one-dimensional tensor."""

    flat = values.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("Cannot compute upper-tail mean of an empty tensor")
    normalized_fraction = float(fraction)
    if not 0.0 < normalized_fraction <= 1.0:
        raise ValueError("tail fraction must lie in (0, 1]")
    count = max(1, math.ceil(flat.numel() * normalized_fraction))
    return torch.topk(flat, k=count, largest=True, sorted=False).values.mean()


def _sequence_language_model_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute one causal-LM cross-entropy per sequence."""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("LM logits/labels must have shapes [B,S,V] and [B,S]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("LM logits and labels must agree in batch/sequence shape")
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].to(device=shift_logits.device)
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(shift_labels.shape)
    valid = shift_labels.ne(-100)
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Each LM sequence must contain at least one valid label")
    return (token_losses * valid).sum(dim=1) / counts


def _freeze_except_replacements(
    model: nn.Module,
    records: Sequence[Any],
) -> list[nn.Parameter]:
    states = _capture_requires_grad_states(model.parameters())
    try:
        for param in model.parameters():
            param.requires_grad_(False)
        trainable: list[nn.Parameter] = []
        seen: set[int] = set()
        for record in records:
            for param in record.replacement.parameters():
                param.requires_grad_(True)
                param_id = id(param)
                if param_id not in seen:
                    seen.add(param_id)
                    trainable.append(param)
        if not trainable:
            raise ValueError("No replacement parameters are trainable")
        return trainable
    except Exception:
        _restore_requires_grad_states(states)
        raise


def _select_joint_trainable_parameters(
    model: nn.Module,
    records: Sequence[Any],
    train_target: str,
) -> list[nn.Parameter]:
    normalized = str(train_target).lower()
    if normalized in {"replacement_only", "replacements_only", "dendritic_only"}:
        return _freeze_except_replacements(model, records)
    if normalized in {"full_model", "all", "student"}:
        params: list[nn.Parameter] = []
        for param in model.parameters():
            param.requires_grad_(True)
            params.append(param)
        if not params:
            raise ValueError("No student parameters are trainable")
        return params
    raise ValueError(
        "joint_lm_distillation train_target must be 'replacement_only' or 'full_model'"
    )


def _resolve_joint_hidden_layers(
    config: Config,
) -> list[int]:
    train_cfg = config.training.transformer_replacement
    configured = list(getattr(train_cfg, "hidden_loss_layers", []) or [])
    if configured:
        return [int(layer) for layer in configured]
    collapsed_spans = list(
        getattr(
            config.model.transformer_replacement,
            "collapsed_replacement_spans",
            [],
        )
        or []
    )
    if collapsed_spans:
        return [int(span[-1]) for span in collapsed_spans]
    return resolve_configured_replacement_layers(config)


def _joint_lm_distillation_loss(
    student_output: Any,
    teacher_output: Any | None,
    *,
    hidden_layers: Sequence[int],
    lm_loss_weight: float,
    kl_loss_weight: float,
    kl_temperature: float,
    hidden_loss_weight: float,
    hidden_loss_type: str = "mse",
    hidden_loss_layer_weights: Sequence[float] | None = None,
    hidden_loss_epsilon: float = 1.0e-6,
    labels: torch.Tensor | None = None,
    sequence_risk_weight: float = 0.0,
    sequence_risk_fraction: float = 0.25,
    teacher_topk_margin_weight: float = 0.0,
    teacher_topk_margin_k: int = 32,
    teacher_topk_margin_epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    if student_output.loss is None:
        raise ValueError("Student output did not include a language-modeling loss")

    risk_weight = float(sequence_risk_weight)
    if not 0.0 <= risk_weight <= 1.0:
        raise ValueError("sequence_risk_weight must lie in [0, 1]")
    lm_mean = student_output.loss
    lm_tail = lm_mean
    lm_objective = lm_mean
    if risk_weight > 0:
        if labels is None:
            raise ValueError("sequence-risk LM loss requires labels")
        per_sequence_lm = _sequence_language_model_losses(
            student_output.logits,
            labels,
        )
        lm_mean = per_sequence_lm.mean()
        lm_tail = _upper_tail_mean(per_sequence_lm, sequence_risk_fraction)
        lm_objective = (1.0 - risk_weight) * lm_mean + risk_weight * lm_tail

    total = lm_objective * float(lm_loss_weight)
    components = {
        "lm": float(lm_objective.detach().float().item()),
        "lm_mean": float(lm_mean.detach().float().item()),
        "lm_cvar": float(lm_tail.detach().float().item()),
        "kl": 0.0,
        "teacher_topk_margin": 0.0,
        "hidden": 0.0,
    }

    if kl_loss_weight > 0:
        if teacher_output is None:
            raise ValueError("KL loss requires teacher_output")
        temperature = max(float(kl_temperature), 1e-6)
        student_logits = student_output.logits[:, :-1, :].float() / temperature
        teacher_logits = teacher_output.logits[:, :-1, :].float() / temperature
        vocab = student_logits.shape[-1]
        student_flat = student_logits.reshape(-1, vocab)
        teacher_flat = teacher_logits.reshape(-1, vocab)
        kl = F.kl_div(
            F.log_softmax(student_flat, dim=-1),
            F.softmax(teacher_flat, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature)
        total = total + float(kl_loss_weight) * kl
        components["kl"] = float(kl.detach().item())

    margin_weight = float(teacher_topk_margin_weight)
    if not math.isfinite(margin_weight) or margin_weight < 0:
        raise ValueError("teacher_topk_margin_weight must be finite and non-negative")
    if margin_weight > 0:
        if teacher_output is None:
            raise ValueError("teacher top-k margin loss requires teacher_output")
        if isinstance(teacher_topk_margin_k, bool) or not isinstance(
            teacher_topk_margin_k, int
        ):
            raise ValueError("teacher_topk_margin_k must be an integer")
        topk = int(teacher_topk_margin_k)
        student_logits = student_output.logits[:, :-1, :].float()
        teacher_logits = teacher_output.logits[:, :-1, :].float()
        if student_logits.shape != teacher_logits.shape:
            raise ValueError("teacher top-k margin logits must have identical shapes")
        vocabulary_size = int(teacher_logits.shape[-1])
        if topk < 2 or topk > vocabulary_size:
            raise ValueError("teacher_topk_margin_k must lie in [2, vocabulary_size]")
        epsilon = float(teacher_topk_margin_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("teacher_topk_margin_epsilon must be finite and positive")

        teacher_selected, teacher_indices = torch.topk(
            teacher_logits.detach(), topk, dim=-1, sorted=True
        )
        student_selected = torch.gather(student_logits, -1, teacher_indices)
        teacher_margins = teacher_selected[..., :1] - teacher_selected[..., 1:]
        student_margins = student_selected[..., :1] - student_selected[..., 1:]
        squared_error = (student_margins - teacher_margins).square()
        teacher_energy = teacher_margins.square()
        if labels is not None:
            if labels.ndim != 2 or labels.shape != student_output.logits.shape[:2]:
                raise ValueError(
                    "labels must match the batch and sequence dimensions of logits"
                )
            valid = labels[:, 1:].ne(-100)
            if not bool(valid.any()):
                raise ValueError("teacher top-k margin loss has no valid tokens")
            squared_error = squared_error[valid]
            teacher_energy = teacher_energy[valid]
        relative_margin_mse = squared_error.mean() / teacher_energy.mean().clamp_min(
            epsilon
        )
        total = total + margin_weight * relative_margin_mse
        components["teacher_topk_margin"] = float(relative_margin_mse.detach().item())

    if hidden_loss_weight > 0:
        if teacher_output is None:
            raise ValueError("Hidden loss requires teacher_output")
        if student_output.hidden_states is None or teacher_output.hidden_states is None:
            raise ValueError("Hidden loss requires output_hidden_states=True")
        loss_type = str(hidden_loss_type).strip().lower()
        supported = {"mse", "relative_mse", "cosine", "relative_mse_cosine"}
        if loss_type not in supported:
            raise ValueError(
                f"hidden_loss_type must be one of {sorted(supported)}, got "
                f"{hidden_loss_type!r}"
            )
        epsilon = float(hidden_loss_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("hidden_loss_epsilon must be finite and positive")
        configured_weights = list(hidden_loss_layer_weights or [])
        if configured_weights and len(configured_weights) != len(hidden_layers):
            raise ValueError(
                "hidden_loss_layer_weights must have one entry per hidden layer"
            )
        weights = (
            [1.0] * len(hidden_layers)
            if not configured_weights
            else [float(value) for value in configured_weights]
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError(
                "hidden_loss_layer_weights must be finite and non-negative"
            )
        weight_sum = sum(weights)
        if hidden_layers and weight_sum <= 0:
            raise ValueError("hidden_loss_layer_weights must contain a positive value")

        hidden_terms = []
        for layer in hidden_layers:
            hidden_index = int(layer) + 1
            student_hidden = student_output.hidden_states[hidden_index].float()
            teacher_hidden = teacher_output.hidden_states[hidden_index].float()
            if student_hidden.shape != teacher_hidden.shape:
                raise ValueError(
                    f"hidden layer {layer} shape mismatch: "
                    f"{tuple(student_hidden.shape)} != {tuple(teacher_hidden.shape)}"
                )
            mse = F.mse_loss(student_hidden, teacher_hidden)
            if loss_type == "mse":
                term = mse
            elif loss_type == "relative_mse":
                teacher_energy = teacher_hidden.square().mean().clamp_min(epsilon)
                term = mse / teacher_energy
            elif loss_type == "cosine":
                term = (
                    1.0
                    - F.cosine_similarity(
                        student_hidden,
                        teacher_hidden,
                        dim=-1,
                        eps=epsilon,
                    )
                ).mean()
            else:
                teacher_energy = teacher_hidden.square().mean().clamp_min(epsilon)
                relative_mse = mse / teacher_energy
                cosine = (
                    1.0
                    - F.cosine_similarity(
                        student_hidden,
                        teacher_hidden,
                        dim=-1,
                        eps=epsilon,
                    )
                ).mean()
                term = 0.5 * (relative_mse + cosine)
            hidden_terms.append(term)
            components[f"hidden_layer_{int(layer)}"] = float(term.detach().item())
        if hidden_terms:
            normalized_weights = (
                torch.tensor(
                    weights,
                    device=hidden_terms[0].device,
                    dtype=hidden_terms[0].dtype,
                )
                / weight_sum
            )
            hidden_loss = torch.sum(torch.stack(hidden_terms) * normalized_weights)
            total = total + float(hidden_loss_weight) * hidden_loss
            components["hidden"] = float(hidden_loss.detach().item())

    components["total"] = float(total.detach().float().item())
    return total, components
