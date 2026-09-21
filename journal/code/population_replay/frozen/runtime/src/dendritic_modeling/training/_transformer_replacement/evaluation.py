"""Evaluation and benchmarking utilities for transformer replacements."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from dendritic_modeling.config import Config
from dendritic_modeling.deployment.ledger import module_storage_ledger
from dendritic_modeling.networks.architectures.transformer import (
    apply_transformer_replacement_config,
    resolve_transformer_layers,
    unwrap_shared_population_replacement,
    validate_collapsed_span_additive_contract,
)
from dendritic_modeling.training._transformer_replacement.builders import (
    _make_synthetic_units,
    _offload_collapsed_span_original_mlps_,
)
from dendritic_modeling.training._transformer_replacement.checkpoints import (
    _load_replacement_record_checkpoints,
)
from dendritic_modeling.training._transformer_replacement.common import (
    DistillationUnit,
    TransformerReplacementBenchmarkResult,
    _compute_distillation_loss,
    _make_tensor_loader,
    _move_replacement_batch_tensors,
    _resolve_device,
    _resolve_dtype,
    _to_plain_mapping,
    resolve_configured_replacement_layers,
)
from dendritic_modeling.training._transformer_replacement.data import (
    _apply_pre_patched_collapsed_spans,
    _apply_pre_patched_replacements,
    _build_hidden_cache_units_and_datasets,
    _build_synthetic_datasets,
    _load_hf_causal_lm_for_joint_training,
    _read_token_source,
)
from dendritic_modeling.training.replacement_common import _write_metrics_json


def _evaluate_tensor_datasets(
    units: Sequence[DistillationUnit],
    datasets: Sequence[TensorDataset],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    loss_name: str,
    cosine_weight: float,
    forward_modules: Sequence[nn.Module] | None = None,
    train_cfg: object | None = None,
) -> float:
    total = 0.0
    count = 0
    modules = (
        list(forward_modules)
        if forward_modules is not None
        else [unit.replacement for unit in units]
    )
    loaders = [
        _make_tensor_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            train_cfg=train_cfg,
            device=device,
        )
        for dataset in datasets
    ]
    with torch.no_grad():
        for module, loader in zip(modules, loaders):
            module.eval()
            for x, target in loader:
                x, target = _move_replacement_batch_tensors(
                    x,
                    target,
                    device=device,
                    dtype=dtype,
                )
                loss = _compute_distillation_loss(
                    module(x),
                    target,
                    loss_name=loss_name,
                    cosine_weight=cosine_weight,
                )
                total += float(loss.item())
                count += 1
    return total / max(count, 1)


def _unique_physical_replacements(items: Sequence[Any]) -> list[nn.Module]:
    """Return replacement parameter owners in first-site order."""

    result: list[nn.Module] = []
    seen: set[int] = set()
    for item in items:
        module = unwrap_shared_population_replacement(item.replacement)
        if id(module) not in seen:
            seen.add(id(module))
            result.append(module)
    return result


def _record_teacher_ffn_modules(record: Any) -> tuple[nn.Module, ...]:
    """Return every dense teacher FFN represented by one placement record."""

    collapsed = tuple(getattr(record, "collapsed_span_original_mlps", ()) or ())
    norms = tuple(getattr(record, "collapsed_span_original_post_mlp_norms", ()) or ())
    return (collapsed + norms) if collapsed else (record.original_mlp,)


def _compute_prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    pred_flat = prediction.reshape(-1, prediction.shape[-1]).float()
    target_flat = target.reshape(-1, target.shape[-1]).float()
    error = pred_flat - target_flat
    mse = float(error.square().mean().item())
    mae = float(error.abs().mean().item())
    target_power = float(target_flat.square().mean().item())
    target_mean = target_flat.mean(dim=0, keepdim=True)
    total_var = float((target_flat - target_mean).square().sum().item())
    sse = float(error.square().sum().item())
    cosine = float(F.cosine_similarity(pred_flat, target_flat, dim=-1).mean().item())
    return {
        "mse": mse,
        "mae": mae,
        "relative_mse": mse / max(target_power, 1e-12),
        "cosine": cosine,
        "r2": 1.0 - sse / max(total_var, 1e-12),
    }


def _prediction_metric_sufficient_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return additive statistics for exact distributed dataset aggregation."""

    if prediction.shape != target.shape or prediction.ndim < 1:
        raise ValueError("prediction and target tensors must have identical shapes")
    prediction_flat = prediction.reshape(-1, prediction.shape[-1]).double()
    target_flat = target.reshape(-1, target.shape[-1]).double()
    if int(prediction_flat.shape[0]) < 1:
        raise ValueError("prediction metrics require at least one row")
    error = prediction_flat - target_flat
    cosine = F.cosine_similarity(prediction_flat, target_flat, dim=-1)
    return {
        "element_count": torch.tensor(float(error.numel()), dtype=torch.float64),
        "row_count": torch.tensor(float(prediction_flat.shape[0]), dtype=torch.float64),
        "squared_error_sum": error.square().sum(),
        "absolute_error_sum": error.abs().sum(),
        "target_square_sum": target_flat.square().sum(),
        "target_feature_sum": target_flat.sum(dim=0),
        "cosine_sum": cosine.sum(),
    }


def _prediction_metrics_from_sufficient_statistics(
    statistics: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute the public prediction metrics from globally summed statistics."""

    element_count = float(statistics["element_count"].item())
    row_count = float(statistics["row_count"].item())
    if element_count <= 0 or row_count <= 0:
        raise ValueError("prediction metric statistics have no observations")
    squared_error_sum = float(statistics["squared_error_sum"].item())
    absolute_error_sum = float(statistics["absolute_error_sum"].item())
    target_square_sum = float(statistics["target_square_sum"].item())
    target_feature_sum = statistics["target_feature_sum"].double()
    total_var = (
        target_square_sum - float(target_feature_sum.square().sum().item()) / row_count
    )
    mse = squared_error_sum / element_count
    target_power = target_square_sum / element_count
    return {
        "mse": mse,
        "mae": absolute_error_sum / element_count,
        "relative_mse": mse / max(target_power, 1e-12),
        "cosine": float(statistics["cosine_sum"].item()) / row_count,
        "r2": 1.0 - squared_error_sum / max(total_var, 1e-12),
    }


def _evaluate_tensor_dataset_metrics(
    units: Sequence[DistillationUnit],
    datasets: Sequence[TensorDataset],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    train_cfg: object | None = None,
) -> dict[str, float]:
    predictions = []
    targets = []
    loaders = [
        _make_tensor_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            train_cfg=train_cfg,
            device=device,
        )
        for dataset in datasets
    ]
    with torch.no_grad():
        for unit, loader in zip(units, loaders):
            unit.replacement.eval()
            for x, target in loader:
                x = x.to(device=device, dtype=dtype)
                predictions.append(unit.replacement(x).detach().cpu())
                targets.append(target.detach().cpu())
    if not predictions:
        raise ValueError("No benchmark batches were evaluated")
    return _compute_prediction_metrics(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
    )


def _synthetic_teacher_parameter_count(config: Config) -> int | None:
    train_cfg = config.training.transformer_replacement
    if str(train_cfg.teacher_source).lower() != "synthetic_mlp":
        return None
    teacher_cfg = _to_plain_mapping(train_cfg.synthetic_teacher)
    hidden_size = int(train_cfg.hidden_size or 32)
    intermediate_size = int(train_cfg.intermediate_size or hidden_size * 4)
    kind = str(teacher_cfg.get("kind", "gated_mlp")).lower()
    bias = bool(teacher_cfg.get("bias", False))
    if kind == "linear":
        count = hidden_size * hidden_size
        return count + (hidden_size if bias else 0)
    if kind == "mlp":
        count = hidden_size * intermediate_size + intermediate_size * hidden_size
        return count + (intermediate_size + hidden_size if bias else 0)
    if kind == "gated_mlp":
        count = 3 * hidden_size * intermediate_size
        return count + (2 * intermediate_size + hidden_size if bias else 0)
    return None


def _evaluate_lm_loss_on_tokens(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    max_batches: int,
    token_offset: int = 0,
    allow_window_wrap: bool = True,
) -> dict[str, Any]:
    """Evaluate causal-LM loss on deterministic windows from ``tokens``."""
    model.eval()
    input_device = next(model.parameters()).device
    token_offset = int(token_offset)
    if token_offset < 0:
        raise ValueError("token_offset must be non-negative")
    fixed_windows = tokens.ndim == 2
    if fixed_windows:
        if int(tokens.shape[1]) != int(sequence_length):
            raise ValueError(
                "Rank-two token windows must match the requested sequence_length"
            )
        if token_offset % int(sequence_length):
            raise ValueError(
                "token_offset must align to sequence_length for rank-two windows"
            )
        if int(tokens.shape[0]) < 1:
            raise ValueError("Token source contains no windows")
        max_start = None
    else:
        if tokens.ndim != 1:
            raise ValueError("Token source must be rank one or rank two")
        max_start = int(tokens.numel()) - int(sequence_length)
        if max_start <= 0:
            raise ValueError("Token source is shorter than sequence_length")

    requested_windows = max(1, int(max_batches)) * int(batch_size)
    required_end = token_offset + requested_windows * int(sequence_length)
    if not allow_window_wrap and required_end > int(tokens.numel()):
        raise ValueError(
            "Non-wrapping evaluation range exceeds the token source: "
            f"offset={token_offset}, windows={requested_windows}, "
            f"sequence_length={sequence_length}, tokens={tokens.numel()}"
        )

    losses = []
    window_losses = []
    window_starts = []
    n_tokens = 0
    with torch.no_grad():
        for batch_idx in range(max(1, int(max_batches))):
            starts = [
                token_offset
                + (batch_idx * int(batch_size) + offset) * int(sequence_length)
                for offset in range(int(batch_size))
            ]
            if fixed_windows:
                row_indices = [start // int(sequence_length) for start in starts]
                if allow_window_wrap:
                    row_indices = [
                        index % int(tokens.shape[0]) for index in row_indices
                    ]
                starts = [index * int(sequence_length) for index in row_indices]
                input_ids = tokens.index_select(
                    0, torch.tensor(row_indices, dtype=torch.long)
                )
            elif allow_window_wrap:
                assert max_start is not None
                starts = [start % max_start for start in starts]
                input_ids = torch.stack(
                    [tokens[start : start + int(sequence_length)] for start in starts]
                )
            else:
                input_ids = torch.stack(
                    [tokens[start : start + int(sequence_length)] for start in starts]
                )
            window_starts.append([int(start) for start in starts])
            input_ids = input_ids.to(device=input_device)
            output = model(input_ids=input_ids, labels=input_ids)
            loss = output.loss.detach().float()
            losses.append(loss.cpu())
            logits = getattr(output, "logits", None)
            if logits is not None:
                # Preserve item-level evidence without materializing another
                # batch-sized FP32 vocabulary tensor for low-precision models.
                window_losses.extend(
                    float(
                        F.cross_entropy(
                            logits[row, :-1, :].float(),
                            input_ids[row, 1:],
                        ).item()
                    )
                    for row in range(int(input_ids.shape[0]))
                )
            n_tokens += int(input_ids.numel())
    mean_loss = float(torch.stack(losses).mean().item())
    loss_tensor = torch.stack(losses)
    perplexity = float(torch.exp(torch.tensor(min(mean_loss, 50.0))).item())
    return {
        "lm_loss": mean_loss,
        "lm_loss_median": float(loss_tensor.median().item()),
        "lm_loss_p90": float(torch.quantile(loss_tensor, 0.90).item()),
        "lm_loss_p95": float(torch.quantile(loss_tensor, 0.95).item()),
        "lm_loss_min": float(loss_tensor.min().item()),
        "lm_loss_max": float(loss_tensor.max().item()),
        "evaluated_windows": float(loss_tensor.numel()),
        "evaluated_batches": int(loss_tensor.numel()),
        "evaluated_sequences": sum(len(starts) for starts in window_starts),
        "per_window_losses_available": len(window_losses) == requested_windows,
        "lm_loss_by_window": (
            window_losses if len(window_losses) == requested_windows else None
        ),
        "window_starts": [start for starts in window_starts for start in starts],
        "lm_loss_by_batch": [float(value.item()) for value in loss_tensor],
        "window_starts_by_batch": window_starts,
        "token_offset": token_offset,
        "allow_window_wrap": bool(allow_window_wrap),
        "requested_token_range": [token_offset, required_end],
        "perplexity": perplexity,
        "n_tokens": float(n_tokens),
    }


def _benchmark_hf_text_replacement(
    config: Config,
    *,
    checkpoint_dir: str,
    device: torch.device,
) -> TransformerReplacementBenchmarkResult:
    """Evaluate original vs patched causal-LM loss on the configured text."""
    train_cfg = config.training.transformer_replacement
    model_cfg = config.model.transformer_replacement
    text_cfg = _to_plain_mapping(train_cfg.text)
    model = _load_hf_causal_lm_for_joint_training(
        config,
        device=device,
        dtype=_resolve_dtype(train_cfg.dtype),
        initialization="pretrained",
    )
    teacher_whole_model_params = sum(param.numel() for param in model.parameters())
    teacher_whole_model_bytes = module_storage_ledger(model)["total_bytes"]
    _, valid_tokens = _read_token_source(config)
    eval_batches = max(
        1, int(train_cfg.valid_samples) // max(1, int(train_cfg.batch_size))
    )
    validation_token_offset = int(text_cfg.get("validation_token_offset", 0) or 0)
    validation_allow_window_wrap = bool(
        text_cfg.get("validation_allow_window_wrap", True)
    )

    original_metrics = _evaluate_lm_loss_on_tokens(
        model,
        valid_tokens,
        batch_size=train_cfg.batch_size,
        sequence_length=train_cfg.sequence_length,
        max_batches=eval_batches,
        token_offset=validation_token_offset,
        allow_window_wrap=validation_allow_window_wrap,
    )

    current_layers = resolve_configured_replacement_layers(config)
    layers = resolve_transformer_layers(model, layers_attr=model_cfg.layers_attr)
    pre_patched_records = _apply_pre_patched_replacements(
        model,
        layers,
        config,
        layer_indices=current_layers,
        device=next(model.parameters()).device,
    )
    pre_patched_collapsed_records = _apply_pre_patched_collapsed_spans(
        model,
        config,
        layer_indices=current_layers,
        device=next(model.parameters()).device,
    )
    current_records = apply_transformer_replacement_config(
        model,
        config.model.transformer_replacement,
        config.model.core,
    )
    records = pre_patched_records + pre_patched_collapsed_records + current_records
    if not records:
        raise ValueError("No transformer replacement records were created")

    _load_replacement_record_checkpoints(
        current_records,
        checkpoint_dir,
        device=next(model.parameters()).device,
    )

    collapsed_records = [
        record for record in records if getattr(record, "collapsed_span_layers", ())
    ]
    if collapsed_records:
        if valid_tokens.ndim == 2:
            if validation_token_offset % int(train_cfg.sequence_length):
                raise ValueError(
                    "validation_token_offset must align to frozen window boundaries"
                )
            start_row = validation_token_offset // int(train_cfg.sequence_length)
            contract_rows = torch.tensor(
                [
                    (start_row + sample_index) % int(valid_tokens.shape[0])
                    for sample_index in range(int(train_cfg.batch_size))
                ],
                dtype=torch.long,
            )
            contract_input_ids = valid_tokens.index_select(0, contract_rows)
        else:
            max_start = int(valid_tokens.numel()) - int(train_cfg.sequence_length)
            contract_starts = [
                (
                    validation_token_offset
                    + sample_index * int(train_cfg.sequence_length)
                )
                % max_start
                for sample_index in range(int(train_cfg.batch_size))
            ]
            contract_input_ids = torch.stack(
                [
                    valid_tokens[start : start + int(train_cfg.sequence_length)]
                    for start in contract_starts
                ]
            )
        contract_input_ids = contract_input_ids.to(
            device=next(model.parameters()).device
        )
        for record in collapsed_records:
            contract = validate_collapsed_span_additive_contract(
                model,
                record,
                contract_input_ids,
                layers_attr=model_cfg.layers_attr,
            )
            cell = getattr(record.replacement, "span_cell", record.replacement)
            diagnostics = dict(getattr(cell, "teacher_topology_diagnostics", {}) or {})
            diagnostics["collapsed_span_additive_contract"] = contract
            cell.teacher_topology_diagnostics = diagnostics
            cell.teacher_topk_diagnostics = diagnostics

    offloaded_span_ffns = _offload_collapsed_span_original_mlps_(records)

    patched_metrics = _evaluate_lm_loss_on_tokens(
        model,
        valid_tokens,
        batch_size=train_cfg.batch_size,
        sequence_length=train_cfg.sequence_length,
        max_batches=eval_batches,
        token_offset=validation_token_offset,
        allow_window_wrap=validation_allow_window_wrap,
    )

    teacher_dense = sum(
        sum(
            param.numel()
            for module in _record_teacher_ffn_modules(record)
            for param in module.parameters()
        )
        for record in records
    )
    physical_replacements = _unique_physical_replacements(records)
    physical_estimates = [
        module.parameter_estimate() for module in physical_replacements
    ]
    logical_estimates = [record.replacement.parameter_estimate() for record in records]
    stored = sum(int(estimate["stored_total"]) for estimate in physical_estimates)
    # A tied core is executed once at every installed FFN site.  Count active
    # terms per logical application; only physical storage is deduplicated.
    active = sum(int(estimate["active_total"]) for estimate in logical_estimates)
    replacement_ledgers = [
        module_storage_ledger(module) for module in physical_replacements
    ]
    replacement_bytes = sum(ledger["total_bytes"] for ledger in replacement_ledgers)
    replacement_index_bytes = sum(
        ledger["index_bytes"] for ledger in replacement_ledgers
    )
    teacher_bytes = sum(
        module_storage_ledger(module)["total_bytes"]
        for record in records
        for module in _record_teacher_ffn_modules(record)
    )
    composed_whole_model_params = sum(param.numel() for param in model.parameters())
    composed_whole_model_bytes = module_storage_ledger(model)["total_bytes"]

    result = TransformerReplacementBenchmarkResult(
        teacher_source="hf_text",
        layer_indices=[int(record.layer_index) for record in records],
        checkpoint_dir=checkpoint_dir,
        initial=original_metrics,
        trained=patched_metrics,
        teacher_dense_params=teacher_dense,
        dendritic_stored_params=stored,
        dendritic_active_params=active,
        stored_reduction_factor=teacher_dense / max(stored, 1),
        active_reduction_factor=teacher_dense / max(active, 1),
        teacher_dense_bytes=teacher_bytes,
        dendritic_stored_bytes=replacement_bytes,
        dendritic_index_bytes=replacement_index_bytes,
        stored_byte_reduction_factor=teacher_bytes / max(replacement_bytes, 1),
        teacher_whole_model_params=teacher_whole_model_params,
        composed_whole_model_stored_params=composed_whole_model_params,
        whole_model_stored_reduction_factor=(
            teacher_whole_model_params / max(composed_whole_model_params, 1)
        ),
        whole_model_parameter_reduction_fraction=(
            1.0 - composed_whole_model_params / max(teacher_whole_model_params, 1)
        ),
        teacher_whole_model_bytes=teacher_whole_model_bytes,
        composed_whole_model_stored_bytes=composed_whole_model_bytes,
        whole_model_stored_byte_reduction_factor=(
            teacher_whole_model_bytes / max(composed_whole_model_bytes, 1)
        ),
        whole_model_byte_reduction_fraction=(
            1.0 - composed_whole_model_bytes / max(teacher_whole_model_bytes, 1)
        ),
        initial_label="original_teacher_model",
        trained_label="trained_replacement_model",
        initialization_diagnostics={
            f"layer_{record.layer_index}": dict(diagnostics)
            for record in records
            if (
                diagnostics := getattr(
                    record.replacement,
                    "teacher_topk_diagnostics",
                    {},
                )
            )
        },
        evaluation_provenance={
            "pre_patched_layer_indices": [
                int(record.layer_index) for record in pre_patched_records
            ],
            "pre_patched_collapsed_spans": [
                list(record.collapsed_span_layers)
                for record in pre_patched_collapsed_records
            ],
            "dataset_name": text_cfg.get("dataset_name"),
            "dataset_config": text_cfg.get("dataset_config"),
            "training_split": text_cfg.get("split"),
            "validation_dataset_name": text_cfg.get("validation_dataset_name"),
            "validation_dataset_config": text_cfg.get("validation_dataset_config"),
            "validation_text_field": text_cfg.get("validation_text_field"),
            "validation_split": text_cfg.get("validation_split"),
            "training_data_files": text_cfg.get("data_files"),
            "validation_data_files": text_cfg.get("validation_data_files"),
            "validation_documents": text_cfg.get("validation_documents"),
            "validation_max_tokens": text_cfg.get("validation_max_tokens"),
            "validation_token_offset": validation_token_offset,
            "validation_allow_window_wrap": validation_allow_window_wrap,
            "sequence_length": int(train_cfg.sequence_length),
            "valid_samples": int(train_cfg.valid_samples),
            "batch_size": int(train_cfg.batch_size),
            "compression_scope_labels": {
                "stored_reduction_factor": "declared_replaced_dense_modules_only",
                "stored_byte_reduction_factor": (
                    "declared_replaced_dense_modules_only"
                ),
                "whole_model_stored_reduction_factor": (
                    "complete_runtime_model_after_checkpoint_load"
                ),
                "whole_model_stored_byte_reduction_factor": (
                    "complete_runtime_model_after_checkpoint_load"
                ),
            },
            **(
                {
                    "logical_replacement_sites": len(records),
                    "physical_replacement_states": len(physical_replacements),
                    "replacement_storage_accounting": (
                        "physical_shared_states_counted_once"
                    ),
                    "replacement_active_compute_accounting": (
                        "active_terms_summed_at_every_ffn_site"
                    ),
                    "parameter_tied_replacement_execution_semantics": (
                        "parameter_tying_not_single_multi_layer_cell"
                    ),
                    "reduces_cell_applications": False,
                }
                if len(physical_replacements) < len(records)
                else {}
            ),
            **(
                {
                    "collapsed_replacement_spans": [
                        list(record.collapsed_span_layers)
                        for record in records
                        if getattr(record, "collapsed_span_layers", ())
                    ],
                    "collapsed_ffn_sites_replaced": sum(
                        len(record.collapsed_span_layers)
                        for record in records
                        if getattr(record, "collapsed_span_layers", ())
                    ),
                    "collapsed_cell_applications": sum(
                        1
                        for record in records
                        if getattr(record, "collapsed_span_layers", ())
                    ),
                    "collapsed_execution_semantics": (
                        "zero_earlier_ffn_branches_one_cell_at_span_exit"
                    ),
                    "collapsed_zero_branch_implementation": (
                        "torch_zeros_like_tensor_write"
                    ),
                    "collapsed_runtime_claim_status": (
                        "requires_measured_family_specific_benchmark"
                    ),
                    "detached_dense_span_ffns_offloaded_to_cpu": int(
                        offloaded_span_ffns
                    ),
                }
                if any(
                    getattr(record, "collapsed_span_layers", ()) for record in records
                )
                else {}
            ),
        },
    )
    return result


def benchmark_saved_transformer_replacement(
    config: Config,
    *,
    checkpoint_dir: str | None = None,
    metrics_filename: str = "benchmark_metrics.json",
) -> TransformerReplacementBenchmarkResult:
    """Compare untrained and trained dendritic replacements against the teacher.

    The benchmark reconstructs the configured teacher data source, evaluates the
    initial dendritic replacement, loads ``layer_<idx>_replacement.pt`` files,
    and evaluates the trained replacement on the same validation hidden states.
    """
    train_cfg = config.training.transformer_replacement
    if not train_cfg.enabled:
        raise ValueError("training.transformer_replacement.enabled must be true")

    device = _resolve_device(train_cfg.device)
    dtype = _resolve_dtype(train_cfg.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    seed = int(train_cfg.seed if train_cfg.seed is not None else config.experiment.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    ckpt_dir = checkpoint_dir or train_cfg.save_dir
    source = str(train_cfg.teacher_source).lower()
    collapsed_spans = list(
        getattr(
            config.model.transformer_replacement,
            "collapsed_replacement_spans",
            [],
        )
        or []
    )
    if collapsed_spans and source != "hf_text":
        raise NotImplementedError(
            "collapsed replacement benchmarking requires the full transformer "
            "trajectory (teacher_source='hf_text')"
        )
    if source == "hf_text":
        result = _benchmark_hf_text_replacement(
            config,
            checkpoint_dir=ckpt_dir,
            device=device,
        )
        result.wall_time_seconds = time.perf_counter() - started
        if device.type == "cuda":
            result.peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated()
            result.peak_cuda_reserved_bytes = torch.cuda.max_memory_reserved()
        _write_metrics_json(
            result,
            output_dir=ckpt_dir,
            filename=metrics_filename,
        )
        return result

    if source == "synthetic_mlp":
        units = _make_synthetic_units(config, device=device, dtype=dtype)
        _, valid_datasets = _build_synthetic_datasets(
            units,
            config,
            device=device,
            dtype=dtype,
        )
    elif source == "hidden_cache":
        units, _, valid_datasets = _build_hidden_cache_units_and_datasets(
            config,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(
            "Post-training benchmark currently supports synthetic_mlp and "
            "hidden_cache teacher sources. Use hidden_cache for large HF models."
        )

    initial_metrics = _evaluate_tensor_dataset_metrics(
        units,
        valid_datasets,
        batch_size=train_cfg.batch_size,
        device=device,
        dtype=dtype,
        train_cfg=train_cfg,
    )

    _load_replacement_record_checkpoints(units, ckpt_dir, device=device)

    trained_metrics = _evaluate_tensor_dataset_metrics(
        units,
        valid_datasets,
        batch_size=train_cfg.batch_size,
        device=device,
        dtype=dtype,
        train_cfg=train_cfg,
    )

    physical_replacements = _unique_physical_replacements(units)
    physical_estimates = [
        module.parameter_estimate() for module in physical_replacements
    ]
    logical_estimates = [unit.replacement.parameter_estimate() for unit in units]
    stored = sum(int(estimate["stored_total"]) for estimate in physical_estimates)
    # A tied core is executed once at every installed FFN site.  Count active
    # terms per logical application; only physical storage is deduplicated.
    active = sum(int(estimate["active_total"]) for estimate in logical_estimates)
    teacher_dense = _synthetic_teacher_parameter_count(config)
    if teacher_dense is not None:
        teacher_dense *= max(len(units), 1)
    replacement_ledgers = [
        module_storage_ledger(module) for module in physical_replacements
    ]
    replacement_bytes = sum(ledger["total_bytes"] for ledger in replacement_ledgers)
    replacement_index_bytes = sum(
        ledger["index_bytes"] for ledger in replacement_ledgers
    )
    teacher_bytes = (
        sum(
            module_storage_ledger(unit.teacher_mlp)["total_bytes"]
            for unit in units
            if unit.teacher_mlp is not None
        )
        or None
    )

    result = TransformerReplacementBenchmarkResult(
        teacher_source=source,
        layer_indices=[unit.layer_index for unit in units],
        checkpoint_dir=ckpt_dir,
        initial=initial_metrics,
        trained=trained_metrics,
        teacher_dense_params=teacher_dense,
        dendritic_stored_params=stored,
        dendritic_active_params=active,
        stored_reduction_factor=(
            None if teacher_dense is None else teacher_dense / max(stored, 1)
        ),
        active_reduction_factor=(
            None if teacher_dense is None else teacher_dense / max(active, 1)
        ),
        teacher_dense_bytes=teacher_bytes,
        dendritic_stored_bytes=replacement_bytes,
        dendritic_index_bytes=replacement_index_bytes,
        stored_byte_reduction_factor=(
            None if teacher_bytes is None else teacher_bytes / max(replacement_bytes, 1)
        ),
        initialization_diagnostics={
            f"layer_{unit.layer_index}": dict(diagnostics)
            for unit in units
            if (
                diagnostics := getattr(
                    unit.replacement,
                    "teacher_topk_diagnostics",
                    {},
                )
            )
        },
        evaluation_provenance=(
            {
                "logical_replacement_sites": len(units),
                "physical_replacement_states": len(physical_replacements),
                "replacement_storage_accounting": (
                    "physical_shared_states_counted_once"
                ),
                "replacement_active_compute_accounting": (
                    "active_terms_summed_at_every_ffn_site"
                ),
                "parameter_tied_replacement_execution_semantics": (
                    "parameter_tying_not_single_multi_layer_cell"
                ),
                "reduces_cell_applications": False,
            }
            if len(physical_replacements) < len(units)
            else {}
        ),
    )

    result.wall_time_seconds = time.perf_counter() - started
    if device.type == "cuda":
        result.peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated()
        result.peak_cuda_reserved_bytes = torch.cuda.max_memory_reserved()

    _write_metrics_json(result, output_dir=ckpt_dir, filename=metrics_filename)
    return result
