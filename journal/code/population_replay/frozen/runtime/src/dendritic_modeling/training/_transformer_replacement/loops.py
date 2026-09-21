"""Training loops for transformer replacement experiments."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import TensorDataset

from dendritic_modeling.config import Config
from dendritic_modeling.config.compression import ModelCompressionConfig
from dendritic_modeling.deployment import freeze_sparse_topology_
from dendritic_modeling.deployment.compression import describe_fixed_sparse_topology
from dendritic_modeling.deployment.ledger import deployment_storage_manifest
from dendritic_modeling.networks.architectures.replacement import (
    initialize_population_topology_from_targets_,
    initialize_population_topology_from_teacher_,
    supports_teacher_topology_initialization,
)
from dendritic_modeling.networks.architectures.replacement.gradient_health import (
    rebalance_positive_pathways_,
)
from dendritic_modeling.networks.architectures.transformer import (
    ZeroFFNResidualBranch,
    apply_transformer_replacement_config,
    resolve_transformer_layers,
    unwrap_shared_population_replacement,
    validate_collapsed_span_additive_contract,
)
from dendritic_modeling.networks.architectures.transformer.utils import (
    _get_attr_path,
    _set_attr_path,
)
from dendritic_modeling.networks.checkpoints import (
    atomic_torch_save_candidates,
    compact_model_state_dict_candidates,
)
from dendritic_modeling.training._transformer_replacement.builders import (
    _make_synthetic_units,
    _offload_collapsed_span_original_mlps_,
    _records_to_units,
)
from dendritic_modeling.training._transformer_replacement.calibration import (
    calibrate_transformer_reactivation,
    calibration_batch_limit,
    prefix_calibration_diagnostics,
    save_reactivation_calibration,
)
from dendritic_modeling.training._transformer_replacement.checkpoints import (
    _load_joint_training_checkpoint,
    _load_layerwise_training_checkpoint,
    _load_replacement_record_checkpoints,
    _restore_best_replacement_checkpoint,
    _save_best_replacement_checkpoint,
    _save_joint_training_checkpoint,
    _save_layerwise_training_checkpoint,
)
from dendritic_modeling.training._transformer_replacement.common import (
    DistillationUnit,
    TransformerReplacementTrainingResult,
    _compute_distillation_loss,
    _make_tensor_loader,
    _move_replacement_batch_tensors,
    _replacement_forward_modules,
    _resolve_device,
    _resolve_dtype,
    resolve_configured_replacement_layers,
)
from dendritic_modeling.training._transformer_replacement.data import (
    _apply_pre_patched_collapsed_spans,
    _apply_pre_patched_replacements,
    _build_hidden_cache_units_and_datasets,
    _build_synthetic_datasets,
    _load_hf_causal_lm_for_joint_training,
    _load_hf_teacher_and_units,
    _MLPIOCapture,
    _read_token_source,
    _sample_token_batch,
)
from dendritic_modeling.training._transformer_replacement.distributed import (
    assert_transformer_distributed_finite_row,
    assert_transformer_distributed_module_state_equal,
    initialize_transformer_distributed,
    is_transformer_main_process,
    transformer_distributed_gather_row,
    transformer_distributed_mean,
    transformer_distributed_sum_count_mean,
    transformer_distributed_sum_tensor,
    transformer_process_rank,
    transformer_process_world_size,
    wrap_transformer_ddp,
)
from dendritic_modeling.training._transformer_replacement.evaluation import (
    _compute_prediction_metrics,
    _evaluate_tensor_datasets,
    _prediction_metric_sufficient_statistics,
    _prediction_metrics_from_sufficient_statistics,
)
from dendritic_modeling.training._transformer_replacement.joint import (
    _joint_lm_distillation_loss,
    _resolve_joint_hidden_layers,
    _select_joint_trainable_parameters,
    _upper_tail_mean,
)
from dendritic_modeling.training._transformer_replacement.model_sources import (
    TransformerModelSourceSession,
)
from dendritic_modeling.training._transformer_replacement.pruning_ladder_runner import (
    run_layerwise_pruning_ladder_,
)
from dendritic_modeling.training._transformer_replacement.recovery_policy import (
    build_joint_recovery_policy,
    validate_joint_recovery_policy_config,
)
from dendritic_modeling.training._transformer_replacement.token_stream import (
    FrozenWindowBatchSource,
    StreamingTokenBatchSource,
)
from dendritic_modeling.training.optimizers import create_optimizer
from dendritic_modeling.training.optimizers.custom import (
    apply_sparse_topology_updates_after_scaled_step,
)
from dendritic_modeling.training.replacement_common import (
    ReplacementTrainingHistory,
    _clip_replacement_grad_norm,
    _infinite_loader,
    _make_cuda_grad_scaler,
    _should_run_periodic_step,
    _write_metrics_json,
)
from dendritic_modeling.training.replacement_pruning import (
    validate_pruning_ladder_config,
)

logger = logging.getLogger(__name__)


def _save_token_stream_state(
    source: torch.Tensor | StreamingTokenBatchSource | FrozenWindowBatchSource,
    save_dir: str,
    *,
    checkpoint: bool = False,
) -> None:
    if isinstance(source, (StreamingTokenBatchSource, FrozenWindowBatchSource)):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        filename = (
            (
                "token_stream_checkpoint_state.json"
                if checkpoint
                else "token_stream_state.json"
            )
            if world_size == 1
            else (
                "token_stream_checkpoint_state" if checkpoint else "token_stream_state"
            )
            + f".rank{transformer_process_rank()}.json"
        )
        source.save_state(os.path.join(save_dir, filename))


def _restore_layerwise_training(
    config: Config,
    units: Sequence[DistillationUnit],
    optimizer: torch.optim.Optimizer,
    scaler,
    *,
    device: torch.device,
) -> tuple[int, ReplacementTrainingHistory | None]:
    resume_path = str(
        getattr(config.training.transformer_replacement, "resume_checkpoint", "") or ""
    )
    if not resume_path:
        return 0, None
    step, history = _load_layerwise_training_checkpoint(
        resume_path,
        units,
        optimizer,
        scaler,
        device=device,
    )
    logger.info("Resumed transformer replacement training at step %s", step)
    return step, history


def _load_layerwise_warm_start(
    config: Config,
    units: Sequence[DistillationUnit],
    *,
    device: torch.device,
) -> bool:
    """Load exported replacement weights for a new layerwise training stage.

    A warm start is intentionally distinct from a resume: it retains the
    learned replacement graph and parameters while rebuilding the optimizer,
    validation history, and data-stream state. This makes recovery and
    curriculum stages auditable as separate runs.
    """

    train_cfg = config.training.transformer_replacement
    warm_start_dir = str(getattr(train_cfg, "warm_start_dir", "") or "")
    resume_path = str(getattr(train_cfg, "resume_checkpoint", "") or "")
    if warm_start_dir and resume_path:
        raise ValueError(
            "warm_start_dir and resume_checkpoint are mutually exclusive: "
            "warm_start_dir begins a new training stage, whereas "
            "resume_checkpoint restores the existing stage"
        )
    if not warm_start_dir:
        return False
    logger.info(
        "Loading layerwise dendritic replacement warm-start from %s",
        warm_start_dir,
    )
    _load_replacement_record_checkpoints(
        units,
        warm_start_dir,
        device=device,
    )
    return True


def _maybe_save_layerwise_training(
    config: Config,
    units: Sequence[DistillationUnit],
    optimizer: torch.optim.Optimizer,
    scaler,
    history: ReplacementTrainingHistory,
    *,
    step: int,
    device: torch.device,
    token_source: (
        torch.Tensor | StreamingTokenBatchSource | FrozenWindowBatchSource | None
    ) = None,
) -> None:
    train_cfg = config.training.transformer_replacement
    every = int(getattr(train_cfg, "checkpoint_every", 0) or 0)
    if every <= 0 or not _should_run_periodic_step(
        step,
        every,
        int(train_cfg.max_steps),
    ):
        return
    if token_source is not None:
        _save_token_stream_state(
            token_source,
            train_cfg.save_dir,
            checkpoint=True,
        )
    _save_layerwise_training_checkpoint(
        os.path.join(train_cfg.save_dir, "training_checkpoint.pt"),
        units,
        optimizer,
        scaler,
        history,
        step=step,
        device=device,
    )


def _save_initial_best_replacement(
    config: Config,
    units: Sequence[DistillationUnit],
    history: ReplacementTrainingHistory,
) -> None:
    train_cfg = config.training.transformer_replacement
    if not bool(getattr(train_cfg, "restore_best_replacement", True)):
        return
    _save_best_replacement_checkpoint(
        train_cfg.save_dir,
        units,
        step=0,
        valid_loss=history.best_valid_loss,
    )


def _record_valid_and_maybe_save_best(
    config: Config,
    units: Sequence[DistillationUnit],
    history: ReplacementTrainingHistory,
    *,
    step: int,
    valid_loss: float,
) -> None:
    previous_best = float(history.best_valid_loss)
    history.record_valid_loss(step, valid_loss)
    train_cfg = config.training.transformer_replacement
    if (
        bool(getattr(train_cfg, "restore_best_replacement", True))
        and float(valid_loss) < previous_best
    ):
        _save_best_replacement_checkpoint(
            train_cfg.save_dir,
            units,
            step=step,
            valid_loss=valid_loss,
        )


def _restore_best_replacement(
    config: Config,
    units: Sequence[DistillationUnit],
    *,
    device: torch.device,
) -> None:
    train_cfg = config.training.transformer_replacement
    if not bool(getattr(train_cfg, "restore_best_replacement", True)):
        return
    restored = _restore_best_replacement_checkpoint(
        train_cfg.save_dir,
        units,
        device=device,
    )
    if restored is not None:
        step, valid_loss = restored
        if step == 0 and int(getattr(train_cfg, "max_steps", 0)) > 0:
            logger.warning(
                "INITIAL CHECKPOINT SELECTED: best_step=0. The restored replacement "
                "is the starting checkpoint; optimizer updates do not establish "
                "a retained recovery improvement. Inspect training history and "
                "report this arm as an initial-checkpoint selection."
            )
        logger.info(
            "Restored validation-best replacement checkpoint step=%s loss=%.6f",
            step,
            valid_loss,
        )


def _physical_replacement_groups(
    items: Sequence[Any],
) -> list[tuple[nn.Module, list[Any]]]:
    """Group records or units by their physical replacement parameter owner."""

    groups: list[tuple[nn.Module, list[Any]]] = []
    positions: dict[int, int] = {}
    for item in items:
        physical = unwrap_shared_population_replacement(item.replacement)
        key = id(physical)
        position = positions.get(key)
        if position is None:
            positions[key] = len(groups)
            groups.append((physical, [item]))
        else:
            groups[position][1].append(item)
    return groups


def _pool_teacher_topology_boundaries(
    units: Sequence[DistillationUnit],
    inputs: dict[int, list[torch.Tensor]],
    *,
    cached_targets: dict[int, list[torch.Tensor]] | None = None,
    max_rows: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Build a deterministic, site-balanced teacher boundary pool."""

    if not units:
        raise ValueError("teacher topology pooling requires at least one unit")
    available_rows = []
    for unit in units:
        batches = inputs.get(int(unit.layer_index), [])
        if not batches:
            raise ValueError(
                f"shared topology pool has no inputs for layer {unit.layer_index}"
            )
        available_rows.append(
            sum(int(batch.numel() // batch.shape[-1]) for batch in batches)
        )
    per_site_limit = min(int(max_rows) // len(units), *available_rows)
    if per_site_limit < 1:
        raise ValueError("shared topology pool has no usable balanced rows")
    pooled_inputs: list[torch.Tensor] = []
    pooled_targets: list[torch.Tensor] = []
    rows_by_layer: dict[str, int] = {}
    for unit in units:
        batches = inputs.get(int(unit.layer_index), [])
        values = torch.cat(batches, dim=0).reshape(-1, batches[0].shape[-1])
        values = values[:per_site_limit]
        if unit.teacher_mlp is None:
            target_batches = (
                []
                if cached_targets is None
                else cached_targets.get(int(unit.layer_index), [])
            )
            if not target_batches:
                raise ValueError(
                    "topology pooling needs either a teacher MLP or cached targets "
                    f"for layer {unit.layer_index}"
                )
            targets = torch.cat(target_batches, dim=0).reshape(
                -1, target_batches[0].shape[-1]
            )[: values.shape[0]]
            if targets.shape[0] != values.shape[0]:
                raise ValueError(
                    "cached teacher targets do not cover the balanced input pool "
                    f"for layer {unit.layer_index}"
                )
        else:
            teacher_parameter = next(unit.teacher_mlp.parameters(), None)
            teacher_device = (
                values.device if teacher_parameter is None else teacher_parameter.device
            )
            teacher_dtype = (
                values.dtype if teacher_parameter is None else teacher_parameter.dtype
            )
            with torch.no_grad():
                targets = unit.teacher_mlp(
                    values.to(device=teacher_device, dtype=teacher_dtype)
                ).detach()
        pooled_inputs.append(values.detach().cpu())
        pooled_targets.append(targets.reshape(-1, targets.shape[-1]).detach().cpu())
        rows_by_layer[str(int(unit.layer_index))] = int(values.shape[0])
    return (
        torch.cat(pooled_inputs, dim=0),
        torch.cat(pooled_targets, dim=0),
        rows_by_layer,
    )


def _initialize_grouped_teacher_topologies(
    units: Sequence[DistillationUnit],
    inputs: dict[int, list[torch.Tensor]],
    *,
    cached_targets: dict[int, list[torch.Tensor]] | None = None,
) -> None:
    """Initialize each physical cell once, pooling all of its teacher sites."""

    for physical, members in _physical_replacement_groups(units):
        if not supports_teacher_topology_initialization(physical):
            continue
        members = sorted(members, key=lambda member: int(member.layer_index))
        if len(members) == 1:
            unit = members[0]
            batches = inputs.get(int(unit.layer_index), [])
            if batches and unit.teacher_mlp is not None:
                initialize_population_topology_from_teacher_(
                    physical,
                    unit.teacher_mlp,
                    torch.cat(batches, dim=0),
                    metric=str(physical.teacher_support_metric),
                )
            elif batches and cached_targets is not None:
                target_batches = cached_targets.get(int(unit.layer_index), [])
                if target_batches:
                    initialize_population_topology_from_targets_(
                        physical,
                        torch.cat(batches, dim=0),
                        torch.cat(target_batches, dim=0),
                        metric=str(physical.teacher_support_metric),
                    )
            continue
        pooled_inputs, pooled_targets, rows_by_layer = (
            _pool_teacher_topology_boundaries(
                members,
                inputs,
                cached_targets=cached_targets,
            )
        )
        diagnostics = initialize_population_topology_from_targets_(
            physical,
            pooled_inputs,
            pooled_targets,
            metric=str(physical.teacher_support_metric),
        )
        diagnostics["parameter_tied_teacher_pool"] = {
            "schema": "dendritic_parameter_tied_teacher_topology_pool/v1",
            "layers": [int(member.layer_index) for member in members],
            "rows_by_layer": rows_by_layer,
            "pooling": "equal_per_site_deterministic_prefix",
        }
        physical.teacher_topology_diagnostics = diagnostics
        physical.teacher_topk_diagnostics = diagnostics


def _calibrate_tensor_replacements(
    units: Sequence[DistillationUnit],
    datasets: Sequence[TensorDataset],
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    train_cfg = config.training.transformer_replacement
    limit = calibration_batch_limit(train_cfg)
    diagnostics = {}
    inputs: dict[int, list[torch.Tensor]] = {}
    targets: dict[int, list[torch.Tensor]] = {}
    for unit, dataset in zip(units, datasets):
        batch_size = max(1, int(train_cfg.batch_size))
        batches = [
            dataset.tensors[0][start : start + batch_size].to(
                device=device,
                dtype=dtype,
            )
            for start in range(0, min(len(dataset), limit * batch_size), batch_size)
        ]
        inputs[int(unit.layer_index)] = batches
        targets[int(unit.layer_index)] = [
            dataset.tensors[1][start : start + batch_size].to(
                device=device,
                dtype=dtype,
            )
            for start in range(0, min(len(dataset), limit * batch_size), batch_size)
        ]
    _initialize_grouped_teacher_topologies(
        units,
        inputs,
        cached_targets=targets,
    )
    for physical, members in _physical_replacement_groups(units):
        members = sorted(members, key=lambda member: int(member.layer_index))
        pooled_batches = [
            batch for member in members for batch in inputs[int(member.layer_index)]
        ]
        current = calibrate_transformer_reactivation(
            physical,
            pooled_batches,
            train_cfg,
            device=device,
        )
        for member in members:
            diagnostics.update(
                prefix_calibration_diagnostics(current, f"layer_{member.layer_index}")
            )
    save_reactivation_calibration(diagnostics, train_cfg.save_dir)


def _reselect_activation_weighted_supports(
    units: Sequence[DistillationUnit],
    inputs: dict[int, list[torch.Tensor]],
) -> None:
    """Re-select TopK supports by attribution-weighted magnitude.

    For replacements requesting ``teacher_topk_metric='activation_weighted'``,
    compute per-feature RMS of the calibration FFN inputs (gate/value scales)
    and of the teacher's own intermediate activations (down-projection scales),
    then re-run teacher initialization so supports maximize |W_ij| * rms(x_j)
    instead of raw weight magnitude. Runs before moment calibration so the
    calibrated output scaling sees the final supports.
    """
    _initialize_grouped_teacher_topologies(units, inputs)
    for unit in units:
        replacement = unit.replacement
        if supports_teacher_topology_initialization(replacement):
            continue
        if getattr(replacement, "teacher_topk_metric", "weight") not in (
            "activation_weighted",
            "activation_weighted_structured",
        ):
            continue
        batches = inputs.get(unit.layer_index)
        if not batches:
            continue
        mlp = unit.teacher_mlp
        weight_device = mlp.gate_proj.weight.device
        weight_dtype = mlp.gate_proj.weight.dtype
        square_in = None
        square_mid = None
        n_rows = 0
        act_fn = getattr(mlp, "act_fn", torch.nn.functional.silu)
        with torch.no_grad():
            for batch in batches:
                flat = batch.reshape(-1, batch.shape[-1]).to(
                    device=weight_device, dtype=weight_dtype
                )
                intermediate = act_fn(mlp.gate_proj(flat)) * mlp.up_proj(flat)
                in_sq = flat.float().square().sum(dim=0)
                mid_sq = intermediate.float().square().sum(dim=0)
                square_in = in_sq if square_in is None else square_in + in_sq
                square_mid = mid_sq if square_mid is None else square_mid + mid_sq
                n_rows += flat.shape[0]
        input_scales = (square_in / max(n_rows, 1)).sqrt()
        intermediate_scales = (square_mid / max(n_rows, 1)).sqrt()
        replacement.initialize_from_mlp(
            mlp,
            input_scales=input_scales,
            intermediate_scales=intermediate_scales,
        )
        logger.info(
            "Layer %d: activation-weighted support re-selection over %d "
            "calibration rows (energies: %s)",
            unit.layer_index,
            n_rows,
            replacement.teacher_topk_diagnostics,
        )


def _calibrate_hf_replacements(
    teacher: nn.Module,
    units: Sequence[DistillationUnit],
    validation_tokens: torch.Tensor,
    capture: _MLPIOCapture,
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    train_cfg = config.training.transformer_replacement
    limit = calibration_batch_limit(train_cfg)
    inputs: dict[int, list[torch.Tensor]] = {unit.layer_index: [] for unit in units}
    input_device = next(teacher.parameters()).device
    with torch.no_grad():
        for _ in range(limit):
            capture.clear()
            input_ids = _sample_token_batch(
                validation_tokens,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                device=input_device,
            )
            teacher(input_ids=input_ids)
            for unit in units:
                inputs[unit.layer_index].append(
                    capture.records[unit.layer_index]["input"].to(
                        device=device,
                        dtype=dtype,
                    )
                )
    _reselect_activation_weighted_supports(units, inputs)
    if bool(getattr(train_cfg, "positive_pathway_rebalance", False)):
        # Softplus pathways initialized in the transform's dead zone receive
        # ~zero gradient (measured 2026-08-24: pre_w means -6.6/-10.9,
        # derivative <=0.013, ~300x less input-path flow than signed cells at
        # the same budget). Re-initialize them inside the live region; the
        # reactivation calibration immediately below restores forward scale.
        for physical, members in _physical_replacement_groups(units):
            unit = min(members, key=lambda member: int(member.layer_index))
            rebalance = rebalance_positive_pathways_(
                physical,
                seed=int(getattr(config.experiment, "seed", 0) or 0)
                + int(unit.layer_index),
            )
            touched = [k for k, v in rebalance.items() if v.get("rebalanced")]
            if touched:
                logger.info(
                    "positive_pathway_rebalance: layer %d moved %d softplus "
                    "pathways into the live region",
                    unit.layer_index,
                    len(touched),
                )
    diagnostics = {}
    for physical, members in _physical_replacement_groups(units):
        members = sorted(members, key=lambda member: int(member.layer_index))
        pooled_batches = [
            batch for member in members for batch in inputs[int(member.layer_index)]
        ]
        current = calibrate_transformer_reactivation(
            physical,
            pooled_batches,
            train_cfg,
            device=device,
        )
        for member in members:
            diagnostics.update(
                prefix_calibration_diagnostics(current, f"layer_{member.layer_index}")
            )
    capture.clear()
    save_reactivation_calibration(diagnostics, train_cfg.save_dir)


def _extract_layer_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError("transformer layer output does not expose a hidden-state tensor")


def _initialize_collapsed_span_exits_from_teacher_(
    student: nn.Module,
    teacher: nn.Module,
    records: Sequence[object],
    validation_tokens: torch.Tensor,
    config: Config,
    *,
    device: torch.device,
) -> set[int]:
    """Fit each span cell to the teacher exit on the collapsed trajectory."""

    collapsed = [record for record in records if record.collapsed_span_layers]
    if not collapsed:
        return set()
    train_cfg = config.training.transformer_replacement
    model_cfg = config.model.transformer_replacement
    student_layers = resolve_transformer_layers(
        student, layers_attr=model_cfg.layers_attr
    )
    teacher_layers = resolve_transformer_layers(
        teacher, layers_attr=model_cfg.layers_attr
    )
    student_training_states = [
        (module, bool(module.training)) for module in student.modules()
    ]
    teacher_training_states = [
        (module, bool(module.training)) for module in teacher.modules()
    ]
    student.eval()
    teacher.eval()
    initialized: set[int] = set()
    try:
        for record in sorted(collapsed, key=lambda item: int(item.layer_index)):
            exit_layer = int(record.layer_index)
            span_layers = tuple(int(layer) for layer in record.collapsed_span_layers)
            student_layer = student_layers[exit_layer]
            teacher_layer = teacher_layers[exit_layer]
            installed = _get_attr_path(student_layer, record.mlp_attr)
            if installed is not record.replacement:
                raise RuntimeError("collapsed span exit changed before calibration")
            probe_input_ids = _sample_token_batch(
                validation_tokens,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                device=device,
            )
            contract = validate_collapsed_span_additive_contract(
                student,
                record,
                probe_input_ids,
                layers_attr=model_cfg.layers_attr,
            )
            zero = ZeroFFNResidualBranch(
                layer_index=exit_layer,
                span_index=int(record.collapsed_span_index),
                span_layers=span_layers,
            )
            cell_inputs: list[torch.Tensor] = []
            student_base_exits: list[torch.Tensor] = []
            teacher_exits: list[torch.Tensor] = []

            def capture_input(_module, args, *, captures=cell_inputs):
                captures.append(args[0].detach())

            def capture_student_exit(
                _module, _args, output, *, captures=student_base_exits
            ):
                captures.append(_extract_layer_hidden(output).detach())

            def capture_teacher_exit(_module, _args, output, *, captures=teacher_exits):
                captures.append(_extract_layer_hidden(output).detach())

            handles = [
                zero.register_forward_pre_hook(capture_input),
                student_layer.register_forward_hook(capture_student_exit),
                teacher_layer.register_forward_hook(capture_teacher_exit),
            ]
            _set_attr_path(student_layer, record.mlp_attr, zero)
            try:
                with torch.no_grad():
                    for _ in range(calibration_batch_limit(train_cfg)):
                        input_ids = _sample_token_batch(
                            validation_tokens,
                            batch_size=train_cfg.batch_size,
                            sequence_length=train_cfg.sequence_length,
                            device=device,
                        )
                        teacher(input_ids=input_ids)
                        student(input_ids=input_ids)
            finally:
                _set_attr_path(student_layer, record.mlp_attr, installed)
                for handle in handles:
                    handle.remove()
            if not cell_inputs or not (
                len(cell_inputs) == len(student_base_exits) == len(teacher_exits)
            ):
                raise RuntimeError("collapsed span calibration capture is incomplete")
            inputs = torch.cat(cell_inputs, dim=0)
            targets = torch.cat(teacher_exits, dim=0) - torch.cat(
                student_base_exits, dim=0
            )
            metric = str(record.replacement.teacher_support_metric)
            diagnostics = (
                initialize_population_topology_from_targets_(
                    record.replacement,
                    inputs,
                    targets,
                    metric=metric,
                )
                if metric
                else {
                    "topology_initialized": False,
                    "reason": "teacher_support_metric_disabled",
                }
            )
            diagnostics["collapsed_span_additive_contract"] = contract
            diagnostics["collapsed_span_exit_target"] = {
                "schema": "dendritic_collapsed_span_exit_target/v1",
                "span_layers": list(span_layers),
                "zero_ffn_layers": list(span_layers[:-1]),
                "exit_layer": exit_layer,
                "target": "teacher_span_exit_minus_zero_cell_student_exit",
                "cell_application_count": 1,
            }
            if record.collapsed_span_post_mlp_norm_attr:
                diagnostics["collapsed_span_exit_target"].update(
                    post_mlp_norm_attr=record.collapsed_span_post_mlp_norm_attr,
                    post_mlp_norm_removed_layers=list(span_layers),
                    cell_output_boundary="post_mlp_norm_residual_branch",
                )
            cell = getattr(record.replacement, "span_cell", record.replacement)
            cell.teacher_topology_diagnostics = diagnostics
            cell.teacher_topk_diagnostics = diagnostics
            initialized.add(exit_layer)
    finally:
        for module, was_training in student_training_states:
            module.training = was_training
        for module, was_training in teacher_training_states:
            module.training = was_training
    return initialized


def _calibrate_joint_replacements(
    student: nn.Module,
    records: Sequence[object],
    validation_tokens: torch.Tensor,
    config: Config,
    *,
    device: torch.device,
    teacher: nn.Module | None = None,
) -> None:
    train_cfg = config.training.transformer_replacement
    collapsed_layers = {
        int(record.layer_index) for record in records if record.collapsed_span_layers
    }
    if collapsed_layers:
        if teacher is None:
            raise ValueError(
                "collapsed span calibration requires the dense teacher model"
            )
        _initialize_collapsed_span_exits_from_teacher_(
            student,
            teacher,
            records,
            validation_tokens,
            config,
            device=device,
        )
    topology_records = [
        record
        for record in records
        if int(record.layer_index) not in collapsed_layers
        if supports_teacher_topology_initialization(record.replacement)
    ]
    if topology_records:
        captured: dict[int, list[torch.Tensor]] = {
            int(record.layer_index): [] for record in topology_records
        }
        handles = []
        for record in topology_records:

            def capture_input(_module, args, *, layer_index=int(record.layer_index)):
                captured[layer_index].append(args[0].detach())

            handles.append(record.replacement.register_forward_pre_hook(capture_input))
        try:
            with torch.no_grad():
                for _ in range(calibration_batch_limit(train_cfg)):
                    input_ids = _sample_token_batch(
                        validation_tokens,
                        batch_size=train_cfg.batch_size,
                        sequence_length=train_cfg.sequence_length,
                        device=device,
                    )
                    student(input_ids=input_ids)
        finally:
            for handle in handles:
                handle.remove()
        topology_units = [
            DistillationUnit(
                layer_index=int(record.layer_index),
                replacement=record.replacement,
                teacher_mlp=record.original_mlp,
            )
            for record in topology_records
        ]
        _initialize_grouped_teacher_topologies(topology_units, captured)

    batches = [
        _sample_token_batch(
            validation_tokens,
            batch_size=train_cfg.batch_size,
            sequence_length=train_cfg.sequence_length,
            device=device,
        )
        for _ in range(calibration_batch_limit(train_cfg))
    ]
    diagnostics = calibrate_transformer_reactivation(
        student,
        batches,
        train_cfg,
        device=device,
    )
    save_reactivation_calibration(diagnostics, train_cfg.save_dir)


def _calibrate_joint_replacements_if_fresh(
    student: nn.Module,
    records: Sequence[object],
    validation_tokens: torch.Tensor,
    config: Config,
    *,
    device: torch.device,
    teacher: nn.Module | None = None,
) -> bool:
    """Calibrate a fresh joint cell without overwriting a learned warm start."""

    train_cfg = config.training.transformer_replacement
    warm_start_dir = str(getattr(train_cfg, "warm_start_dir", "") or "")
    resume_checkpoint = str(getattr(train_cfg, "resume_checkpoint", "") or "")
    if warm_start_dir or resume_checkpoint:
        train_cfg = config.training.transformer_replacement
        model_cfg = config.model.transformer_replacement
        for record in records:
            if not record.collapsed_span_layers:
                continue
            probe_input_ids = _sample_token_batch(
                validation_tokens,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                device=device,
            )
            contract = validate_collapsed_span_additive_contract(
                student,
                record,
                probe_input_ids,
                layers_attr=model_cfg.layers_attr,
            )
            cell = getattr(record.replacement, "span_cell", record.replacement)
            diagnostics = dict(getattr(cell, "teacher_topology_diagnostics", {}) or {})
            diagnostics["collapsed_span_additive_contract"] = contract
            cell.teacher_topology_diagnostics = diagnostics
            cell.teacher_topk_diagnostics = diagnostics
        logger.info(
            "Preserving warm-started or resumed topology, affine paths, and "
            "reactivation parameters after validating collapsed-span contracts; "
            "skipping fresh joint replacement calibration"
        )
        return False
    _calibrate_joint_replacements(
        student,
        records,
        validation_tokens,
        config,
        device=device,
        teacher=teacher,
    )
    return True


def _save_result(
    result: TransformerReplacementTrainingResult,
    units: Sequence[DistillationUnit],
    *,
    save_replacements: bool,
    topology_encoding: str = "uint",
    freeze_sparse_topology_on_export: bool = False,
    describe_fixed_topology: bool = False,
    stochastic_topology_freeze_policy: str = "reject",
    stochastic_topology_freeze_seed: int = 0,
    ragged_topology_format: str = "csr",
) -> None:
    if not is_transformer_main_process():
        return
    _write_metrics_json(result)
    initialization_diagnostics = {
        f"layer_{unit.layer_index}": dict(diagnostics)
        for unit in units
        if (
            diagnostics := getattr(
                unit.replacement,
                "teacher_topk_diagnostics",
                {},
            )
        )
    }
    if initialization_diagnostics:
        with open(
            os.path.join(result.save_dir, "teacher_topk_initialization.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(initialization_diagnostics, handle, indent=2)
            handle.write("\n")
    teacher_ground_truth = {
        f"layer_{unit.layer_index}": ground_truth
        for unit in units
        if unit.teacher_mlp is not None
        and callable(getattr(unit.teacher_mlp, "ground_truth", None))
        and (ground_truth := unit.teacher_mlp.ground_truth()) is not None
    }
    if teacher_ground_truth:
        with open(
            os.path.join(result.save_dir, "teacher_ground_truth.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(teacher_ground_truth, handle, indent=2)
            handle.write("\n")
    if not save_replacements:
        return
    normalized_encoding = str(topology_encoding).strip().lower()
    if normalized_encoding not in {"auto", "uint", "bitmask"}:
        raise ValueError(
            "replacement_checkpoint_encoding must be 'auto', 'uint', or 'bitmask'"
        )
    physical_groups = _physical_replacement_groups(units)
    members_by_physical_id = {
        id(physical): members for physical, members in physical_groups
    }
    has_parameter_tied_replacements = any(
        len(members) > 1 for _physical, members in physical_groups
    )
    collapsed_units = [unit for unit in units if unit.collapsed_span_layers]
    has_collapsed_replacements = bool(collapsed_units)
    if has_parameter_tied_replacements and has_collapsed_replacements:
        raise RuntimeError(
            "parameter-tied groups and collapsed spans cannot share one export"
        )
    exported_physical_ids: set[int] = set()
    artifact_records: list[dict[str, object]] = []
    for unit in units:
        physical = unwrap_shared_population_replacement(unit.replacement)
        physical_id = id(physical)
        members = members_by_physical_id[physical_id]
        if physical_id in exported_physical_ids:
            continue
        exported_physical_ids.add(physical_id)
        leader_layer = getattr(unit, "tied_group_leader", None)
        if len(members) > 1 and leader_layer is None:
            raise RuntimeError(
                "shared physical replacement is missing explicit parameter-tying "
                "group metadata"
            )
        if leader_layer is None:
            export_unit = unit
            alias_layers: list[int] = []
        else:
            leader_layer = int(leader_layer)
            export_unit = next(
                member for member in members if int(member.layer_index) == leader_layer
            )
            alias_layers = [
                int(member.layer_index)
                for member in members
                if int(member.layer_index) != leader_layer
            ]
        tied_manifest = (
            {
                "schema": "dendritic_parameter_tied_replacement_group/v1",
                "leader_layer": int(export_unit.layer_index),
                "alias_layers": sorted(alias_layers),
                "application_layers": sorted(
                    [int(export_unit.layer_index), *alias_layers]
                ),
                "application_count": len(members),
                "physical_state_count": 1,
                "execution_semantics": "shared_cell_applied_at_every_ffn_site",
                "reduces_cell_applications": False,
            }
            if alias_layers
            else None
        )
        collapsed_manifest = (
            {
                "schema": "dendritic_collapsed_replacement_span/v1",
                "span_index": int(export_unit.collapsed_span_index),
                "span_layers": list(export_unit.collapsed_span_layers),
                "zero_ffn_layers": list(export_unit.collapsed_span_layers[:-1]),
                "exit_layer": int(export_unit.layer_index),
                "physical_state_count": 1,
                "cell_application_count": 1,
                "execution_semantics": "one_cell_applied_only_at_span_exit",
            }
            if export_unit.collapsed_span_layers
            else None
        )
        if collapsed_manifest and export_unit.collapsed_span_post_mlp_norm_attr:
            collapsed_manifest.update(
                post_mlp_norm_attr=export_unit.collapsed_span_post_mlp_norm_attr,
                post_mlp_norm_removed_layers=list(export_unit.collapsed_span_layers),
                cell_output_boundary="post_mlp_norm_residual_branch",
                removed_post_mlp_norm_parameters=sum(
                    parameter.numel()
                    for norm in export_unit.collapsed_span_original_post_mlp_norms
                    for parameter in norm.parameters()
                ),
            )
        export_replacement = physical if alias_layers else export_unit.replacement
        topology_manifest = [
            dict(record)
            for record in getattr(
                export_replacement,
                "_loaded_sparse_topology_manifest",
                [],
            )
        ]
        if freeze_sparse_topology_on_export:
            export_replacement.eval()
            records = freeze_sparse_topology_(
                export_replacement,
                ModelCompressionConfig(
                    sparse_topology={
                        "topology_encoding": normalized_encoding,
                        "stochastic_topk_policy": stochastic_topology_freeze_policy,
                        "stochastic_seed": int(stochastic_topology_freeze_seed),
                        "ragged_topology_format": ragged_topology_format,
                    }
                ),
            )
            converted = [dataclasses.asdict(record) for record in records]
            topology_manifest = list(
                {
                    str(record["path"]): record
                    for record in topology_manifest + converted
                }.values()
            )
        if describe_fixed_topology:
            # A structural operation (pruning-ladder rung) changed shapes
            # after construction; record every already-fixed indexed module
            # so the export can be reloaded without the source config's
            # original contact counts.
            described = [
                dataclasses.asdict(record)
                for record in describe_fixed_sparse_topology(export_replacement)
            ]
            covered = {str(record["path"]) for record in topology_manifest}
            topology_manifest.extend(
                record for record in described if str(record["path"]) not in covered
            )
        state_candidates = compact_model_state_dict_candidates(
            export_replacement.state_dict(),
            topology_encoding=normalized_encoding,
        )
        payloads = {
            encoding: {
                "schema_version": (
                    5 if collapsed_manifest else (4 if tied_manifest else 3)
                ),
                "layer_index": export_unit.layer_index,
                "state_dict": state_dict,
                "sparse_index_encoding": index_encoding,
                "sparse_topology_manifest": topology_manifest,
                "topology_encoding_requested": normalized_encoding,
                "topology_encoding_selected": encoding,
                "parameter_estimate": export_replacement.parameter_estimate(),
                "teacher_topk_initialization": getattr(
                    export_replacement,
                    "teacher_topk_diagnostics",
                    {},
                ),
                "selection_manifest": getattr(
                    export_unit.replacement,
                    "selection_manifest",
                    {},
                ),
                "compiled_replacement_plan": getattr(
                    export_replacement,
                    "compiled_replacement_plan",
                    {},
                ),
                "deployment_storage_manifest": deployment_storage_manifest(
                    export_replacement,
                    state_dict,
                    topology_encoding=encoding,
                    sparse_topology_manifest=topology_manifest,
                    scope=(
                        f"transformer_collapsed_span_{export_unit.layer_index}_replacement"
                        if collapsed_manifest
                        else (
                            f"transformer_parameter_tied_group_{export_unit.layer_index}_replacement"
                            if tied_manifest
                            else f"transformer_layer_{export_unit.layer_index}_replacement"
                        )
                    ),
                ),
                **(
                    {"parameter_tied_replacement": tied_manifest}
                    if tied_manifest
                    else {}
                ),
                **(
                    {"collapsed_replacement_span": collapsed_manifest}
                    if collapsed_manifest
                    else {}
                ),
                "teacher_ground_truth": (
                    export_unit.teacher_mlp.ground_truth()
                    if export_unit.teacher_mlp is not None
                    and callable(getattr(export_unit.teacher_mlp, "ground_truth", None))
                    else None
                ),
            }
            for encoding, (state_dict, index_encoding) in state_candidates.items()
        }
        checkpoint_path = os.path.join(
            result.save_dir,
            f"layer_{export_unit.layer_index}_replacement.pt",
        )
        encoding_report = atomic_torch_save_candidates(payloads, checkpoint_path)
        artifact_records.append(
            {
                "layer_index": int(export_unit.layer_index),
                "checkpoint": os.path.basename(checkpoint_path),
                "compact_checkpoint_bytes": int(encoding_report["output_bytes"]),
                "sha256": str(encoding_report["output_sha256"]),
                "topology_encoding_requested": normalized_encoding,
                "topology_encoding_selected": str(encoding_report["selected"]),
                "topology_encoding_candidate_bytes": dict(
                    encoding_report["candidate_bytes"]
                ),
                "selection_basis": str(encoding_report["selection_basis"]),
                **({"alias_layers": sorted(alias_layers)} if alias_layers else {}),
                **(
                    {
                        "collapsed_span_layers": list(
                            export_unit.collapsed_span_layers
                        ),
                        "zero_ffn_layers": list(export_unit.collapsed_span_layers[:-1]),
                        "cell_application_count": 1,
                        **{
                            key: value
                            for key, value in collapsed_manifest.items()
                            if key
                            in {
                                "post_mlp_norm_attr",
                                "post_mlp_norm_removed_layers",
                                "cell_output_boundary",
                                "removed_post_mlp_norm_parameters",
                            }
                        },
                    }
                    if collapsed_manifest
                    else {}
                ),
            }
        )
    export_manifest = {
        "schema": (
            "dendritic_replacement_artifact_manifest/v3"
            if has_collapsed_replacements
            else (
                "dendritic_replacement_artifact_manifest/v2"
                if has_parameter_tied_replacements
                else "dendritic_replacement_artifact_manifest/v1"
            )
        ),
        "scope": "transformer_replacement_layers",
        "artifacts": artifact_records,
        "compact_checkpoint_bytes": int(
            sum(record["compact_checkpoint_bytes"] for record in artifact_records)
        ),
        **(
            {
                "logical_replacement_sites": len(units),
                "physical_replacement_states": len(physical_groups),
                "storage_accounting": "physical_shared_states_counted_once",
                "unique_stored_parameter_values": int(
                    sum(
                        int(physical.parameter_estimate()["stored_total"])
                        for physical, _members in physical_groups
                    )
                ),
                "physical_core_active_parameter_terms": int(
                    sum(
                        int(physical.parameter_estimate()["active_total"])
                        for physical, _members in physical_groups
                    )
                ),
                "logical_site_active_parameter_terms": int(
                    sum(
                        int(unit.replacement.parameter_estimate()["active_total"])
                        for unit in units
                    )
                ),
                "active_compute_accounting": (
                    "active_parameter_terms_summed_at_every_ffn_site; "
                    "this is an operation-count proxy, not measured latency"
                ),
                "parameter_tied_replacement_execution_semantics": (
                    "parameter_tying_not_single_multi_layer_cell"
                ),
                "reduces_cell_applications": False,
            }
            if has_parameter_tied_replacements
            else {}
        ),
        **(
            {
                "collapsed_replacement_spans": [
                    list(unit.collapsed_span_layers) for unit in collapsed_units
                ],
                "teacher_ffn_sites_replaced": int(
                    sum(len(unit.collapsed_span_layers) for unit in collapsed_units)
                ),
                "physical_replacement_states": len(collapsed_units),
                "cell_application_count_per_token": len(collapsed_units),
                "unique_stored_parameter_values": int(
                    sum(
                        int(unit.replacement.parameter_estimate()["stored_total"])
                        for unit in collapsed_units
                    )
                ),
                "active_parameter_terms_per_token": int(
                    sum(
                        int(unit.replacement.parameter_estimate()["active_total"])
                        for unit in collapsed_units
                    )
                ),
                "storage_accounting": "one_physical_cell_per_collapsed_span",
                "active_compute_accounting": (
                    "one_cell_application_at_each_collapsed_span_exit; active "
                    "parameter terms are an operation-count proxy, not latency"
                ),
                "collapsed_execution_semantics": (
                    "zero_earlier_ffn_branches_one_cell_at_span_exit"
                ),
                "zero_branch_implementation": "torch_zeros_like_tensor_write",
                "runtime_claim_status": ("requires_measured_family_specific_benchmark"),
                "staged_add_span_composition": (
                    "supported_via_hash_pinned_frozen_pre_patched_collapsed_spans"
                ),
            }
            if has_collapsed_replacements
            else {}
        ),
    }
    with open(
        os.path.join(result.save_dir, "replacement_export_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(export_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_student_model(
    student: nn.Module,
    config: Config,
) -> None:
    """Save an opt-in reconstructible full student state dict."""
    train_cfg = config.training.transformer_replacement
    if not is_transformer_main_process() or not bool(
        getattr(train_cfg, "save_student_model", False)
    ):
        return
    model_cfg = config.model.transformer_replacement
    if list(getattr(model_cfg, "parameter_tied_replacement_groups", []) or []):
        raise NotImplementedError(
            "save_student_model is not alias-aware for parameter-tied replacement "
            "groups; "
            "use the deduplicated replacement export manifest"
        )
    torch.save(
        {
            "state_dict": student.state_dict(),
            "model_name": str(model_cfg.model_name),
            "model_source": dict(getattr(model_cfg, "model_source", {}) or {}),
            "model_source_receipt": getattr(
                student, "_dendritic_model_source_receipt", None
            ),
            "replacement_layers": resolve_configured_replacement_layers(config),
            "target_module": str(model_cfg.target_module),
            "student_initialization": str(train_cfg.student_initialization),
            "reconstruction": (
                "Load the provenance-bound base declared by the same experiment "
                "YAML, apply its replacement config, then load this state_dict."
            ),
        },
        os.path.join(train_cfg.save_dir, "student_model.pt"),
    )


def _train_tensor_dataset_source(
    units: Sequence[DistillationUnit],
    train_datasets: Sequence[TensorDataset],
    valid_datasets: Sequence[TensorDataset],
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformerReplacementTrainingResult:
    train_cfg = config.training.transformer_replacement
    warm_started = _load_layerwise_warm_start(
        config,
        units,
        device=device,
    )
    if not warm_started and not str(getattr(train_cfg, "resume_checkpoint", "") or ""):
        _calibrate_tensor_replacements(
            units,
            train_datasets,
            config,
            device=device,
            dtype=dtype,
        )
    loaders = [
        _infinite_loader(
            _make_tensor_loader(
                dataset,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                train_cfg=train_cfg,
                device=device,
            )
        )
        for dataset in train_datasets
    ]
    forward_modules = _replacement_forward_modules(units, train_cfg, device)
    trainable_modules = nn.ModuleList(forward_modules)
    optimizer = create_optimizer(
        trainable_modules.parameters(), config.training.main.optimizer
    )
    scaler = _make_cuda_grad_scaler(train_cfg.use_amp, device)

    start_step, history = _restore_layerwise_training(
        config,
        units,
        optimizer,
        scaler,
        device=device,
    )
    if history is None:
        initial_valid = _evaluate_tensor_datasets(
            units,
            valid_datasets,
            batch_size=train_cfg.batch_size,
            device=device,
            dtype=dtype,
            loss_name=train_cfg.loss,
            cosine_weight=train_cfg.cosine_weight,
            forward_modules=forward_modules,
            train_cfg=train_cfg,
        )
        initial_valid = transformer_distributed_mean(initial_valid, device)
        history = ReplacementTrainingHistory.start(initial_valid)
        _save_initial_best_replacement(config, units, history)

    for step in range(start_step + 1, int(train_cfg.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device, dtype=dtype)
        for module, loader in zip(forward_modules, loaders):
            module.train()
            x, target = next(loader)
            x, target = _move_replacement_batch_tensors(
                x,
                target,
                device=device,
                dtype=dtype,
            )
            with autocast("cuda", enabled=scaler.is_enabled()):
                prediction = module(x)
                loss = _compute_distillation_loss(
                    prediction,
                    target,
                    loss_name=train_cfg.loss,
                    cosine_weight=train_cfg.cosine_weight,
                )
            step_loss = step_loss + loss / max(len(units), 1)
        scaler.scale(step_loss).backward()
        _clip_replacement_grad_norm(
            trainable_modules.parameters(),
            max_grad_norm=getattr(train_cfg, "max_grad_norm", None),
            scaler=scaler,
            optimizer=optimizer,
        )
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        apply_sparse_topology_updates_after_scaled_step(
            trainable_modules,
            optimizer,
            scaler,
            scale_before=scale_before,
        )
        history.record_train_loss(
            transformer_distributed_mean(
                float(step_loss.detach().item()),
                device,
            )
        )

        if _should_run_periodic_step(step, train_cfg.eval_every, train_cfg.max_steps):
            valid = _evaluate_tensor_datasets(
                units,
                valid_datasets,
                batch_size=train_cfg.batch_size,
                device=device,
                dtype=dtype,
                loss_name=train_cfg.loss,
                cosine_weight=train_cfg.cosine_weight,
                forward_modules=forward_modules,
                train_cfg=train_cfg,
            )
            valid = transformer_distributed_mean(valid, device)
            _record_valid_and_maybe_save_best(
                config,
                units,
                history,
                step=step,
                valid_loss=valid,
            )
        if _should_run_periodic_step(step, train_cfg.log_every, 1):
            logger.info(
                "transformer replacement distillation step=%s/%s train_loss=%.6f valid_loss=%.6f",
                step,
                train_cfg.max_steps,
                history.train_losses[-1],
                history.valid_losses[-1],
            )
        _maybe_save_layerwise_training(
            config,
            units,
            optimizer,
            scaler,
            history,
            step=step,
            device=device,
        )

    _restore_best_replacement(config, units, device=device)
    result = TransformerReplacementTrainingResult(
        **history.as_result_kwargs(),
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
        execution_mode=(
            "trained" if len(history.train_losses) > 0 else "zero_shot_compile"
        ),
        steps_requested=int(train_cfg.max_steps),
        steps_executed=len(history.train_losses),
    )
    _save_result(
        result,
        units,
        save_replacements=train_cfg.save_replacements,
        topology_encoding=str(train_cfg.replacement_checkpoint_encoding),
        freeze_sparse_topology_on_export=bool(
            getattr(train_cfg, "freeze_sparse_topology_on_export", False)
        ),
        stochastic_topology_freeze_policy=str(
            getattr(train_cfg, "stochastic_topology_freeze_policy", "reject")
        ),
        stochastic_topology_freeze_seed=int(
            getattr(train_cfg, "stochastic_topology_freeze_seed", 0)
        ),
        ragged_topology_format=str(getattr(train_cfg, "ragged_topology_format", "csr")),
    )
    return result


def _uses_trajectory_target(config: Config) -> bool:
    return (
        str(
            getattr(
                config.training.transformer_replacement,
                "distillation_target",
                "teacher_function",
            )
        )
        .strip()
        .lower()
        == "trajectory_anchored"
    )


def _capture_hf_distillation_pairs(
    teacher: nn.Module,
    units: Sequence[DistillationUnit],
    capture: _MLPIOCapture,
    input_ids: torch.Tensor,
    *,
    trajectory: bool,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Forward one token batch and return per-layer (input, target) pairs.

    ``teacher_function``: one forward of the (possibly pre-patched) teacher;
    targets are the dense layer's outputs on the captured inputs.
    ``trajectory_anchored``: a second, clean-teacher forward anchors targets
    to the clean trajectory — target = clean_out + clean_hidden -
    drifted_hidden — so replacements compensate accumulated upstream drift
    (first-order in the block's post-norm). With nothing pre-patched the two
    forwards coincide and this reduces to ``teacher_function``.
    """
    capture.clear()
    with torch.no_grad():
        teacher(input_ids=input_ids)
    swaps = getattr(teacher, "_dendritic_prepatch_swaps", None) or []
    if not getattr(_capture_hf_distillation_pairs, "_logged", False):
        _capture_hf_distillation_pairs._logged = True
        logger.info(
            "distillation pairs: trajectory=%s prepatch_swaps=%d",
            trajectory,
            len(swaps),
        )
    if not trajectory or not swaps:
        return {
            unit.layer_index: (
                capture.records[unit.layer_index]["input"],
                capture.records[unit.layer_index]["target"],
            )
            for unit in units
        }
    drifted = {
        unit.layer_index: {
            "input": capture.records[unit.layer_index]["input"],
            "hidden": capture.records[unit.layer_index]["hidden"],
        }
        for unit in units
    }
    for parent, attr, original, _replacement in swaps:
        setattr(parent, attr, original)
    capture.clear()
    with torch.no_grad():
        teacher(input_ids=input_ids)
    pairs = {}
    for unit in units:
        clean = capture.records[unit.layer_index]
        correction = (
            clean["hidden"].float() - drifted[unit.layer_index]["hidden"].float()
        )
        target = clean["target"].float() + correction
        if not getattr(_capture_hf_distillation_pairs, "_drift_logged", False):
            hidden_norm = clean["hidden"].float().norm().clamp_min(1e-12)
            logger.info(
                "trajectory drift layer %d: ||correction||/||hidden|| = %.6f",
                unit.layer_index,
                float(correction.norm() / hidden_norm),
            )
        pairs[unit.layer_index] = (drifted[unit.layer_index]["input"], target)
    _capture_hf_distillation_pairs._drift_logged = True
    for parent, attr, _original, replacement in swaps:
        setattr(parent, attr, replacement)
    return pairs


def _evaluate_hf_text(
    teacher: nn.Module,
    units: Sequence[DistillationUnit],
    tokens: torch.Tensor,
    capture: _MLPIOCapture,
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_batches: int = 4,
    forward_modules: Sequence[nn.Module] | None = None,
) -> float:
    train_cfg = config.training.transformer_replacement
    input_device = next(teacher.parameters()).device
    modules = (
        list(forward_modules)
        if forward_modules is not None
        else [unit.replacement for unit in units]
    )
    trajectory = _uses_trajectory_target(config)
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_index in range(max(1, n_batches)):
            input_ids = _deterministic_validation_token_batch(
                tokens,
                batch_index=batch_index,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                device=input_device,
            )
            pairs = _capture_hf_distillation_pairs(
                teacher, units, capture, input_ids, trajectory=trajectory
            )
            for unit, module in zip(units, modules):
                x, target = pairs[unit.layer_index]
                x = x.to(device=device, dtype=dtype)
                # fp32 targets preserve the trajectory correction, which can
                # round away entirely under bf16.
                target = target.to(device=device, dtype=torch.float32)
                module.eval()
                loss = _compute_distillation_loss(
                    module(x),
                    target,
                    loss_name=train_cfg.loss,
                    cosine_weight=train_cfg.cosine_weight,
                )
                total += float(loss.item())
                count += 1
    return transformer_distributed_sum_count_mean(total, count, device)


def _deterministic_validation_token_batch(
    tokens: torch.Tensor,
    *,
    batch_index: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    rank: int | None = None,
    world_size: int | None = None,
) -> torch.Tensor:
    """Return a deterministic, rank-strided shard of one global batch.

    Interleaving the rows from ranks ``0..world_size-1`` reconstructs exactly
    the world-one batch with ``batch_size * world_size`` rows.  World-one calls
    preserve the historical indexing convention exactly.
    """

    resolved_rank = transformer_process_rank() if rank is None else int(rank)
    resolved_world_size = (
        transformer_process_world_size() if world_size is None else int(world_size)
    )
    if (
        resolved_world_size < 1
        or resolved_rank < 0
        or resolved_rank >= resolved_world_size
    ):
        raise ValueError(
            f"invalid validation rank/world_size: {resolved_rank}/{resolved_world_size}"
        )
    global_batch_start = int(batch_index) * int(batch_size) * resolved_world_size
    global_indices = [
        global_batch_start + offset * resolved_world_size + resolved_rank
        for offset in range(int(batch_size))
    ]
    if tokens.ndim == 2:
        if int(tokens.shape[1]) != int(sequence_length):
            raise ValueError("Rank-two validation windows must match sequence_length")
        indices = torch.tensor(
            [index % int(tokens.shape[0]) for index in global_indices],
            dtype=torch.long,
        )
        return tokens.index_select(0, indices).to(device=device)
    if tokens.ndim != 1:
        raise ValueError("Validation token source must be rank one or rank two")
    max_start = int(tokens.numel()) - int(sequence_length)
    if max_start <= 0:
        raise ValueError("Validation token source is shorter than sequence_length")
    starts = [index * int(sequence_length) for index in global_indices]
    return torch.stack(
        [
            tokens[start % max_start : start % max_start + sequence_length]
            for start in starts
        ]
    ).to(device=device)


def _evaluate_hf_text_prediction_metrics(
    teacher: nn.Module,
    units: Sequence[DistillationUnit],
    tokens: torch.Tensor,
    capture: _MLPIOCapture,
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_batches: int,
    forward_modules: Sequence[nn.Module] | None = None,
) -> dict[str, dict[str, float]]:
    """Measure held-out layer predictions on deterministic token windows."""
    train_cfg = config.training.transformer_replacement
    input_device = next(teacher.parameters()).device
    modules = (
        list(forward_modules)
        if forward_modules is not None
        else [unit.replacement for unit in units]
    )
    predictions: dict[int, list[torch.Tensor]] = {
        unit.layer_index: [] for unit in units
    }
    targets: dict[int, list[torch.Tensor]] = {unit.layer_index: [] for unit in units}
    trajectory = _uses_trajectory_target(config)
    with torch.no_grad():
        for batch_index in range(max(1, int(n_batches))):
            input_ids = _deterministic_validation_token_batch(
                tokens,
                batch_index=batch_index,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                device=input_device,
            )
            pairs = _capture_hf_distillation_pairs(
                teacher, units, capture, input_ids, trajectory=trajectory
            )
            for unit, module in zip(units, modules):
                x, target = pairs[unit.layer_index]
                x = x.to(device=device, dtype=dtype)
                # fp32 targets preserve the trajectory correction, which can
                # round away entirely under bf16.
                target = target.to(device=device, dtype=torch.float32)
                module.eval()
                predictions[unit.layer_index].append(module(x).detach().cpu())
                targets[unit.layer_index].append(target.detach().cpu())

    result = {}
    for unit in units:
        layer_index = unit.layer_index
        prediction = torch.cat(predictions[layer_index], dim=0)
        target = torch.cat(targets[layer_index], dim=0)
        if transformer_process_world_size() == 1:
            # Preserve the historical world-one arithmetic exactly.
            metrics = _compute_prediction_metrics(prediction, target)
            n_tokens = float(prediction.numel() // prediction.shape[-1])
        else:
            local_statistics = _prediction_metric_sufficient_statistics(
                prediction,
                target,
            )
            global_statistics = {
                name: transformer_distributed_sum_tensor(value, device)
                for name, value in local_statistics.items()
            }
            metrics = _prediction_metrics_from_sufficient_statistics(global_statistics)
            n_tokens = float(global_statistics["row_count"].item())
        metrics["n_tokens"] = n_tokens
        result[f"layer_{layer_index}"] = metrics
    return result


def _train_hf_text_source(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformerReplacementTrainingResult:
    train_cfg = config.training.transformer_replacement
    model_cfg = config.model.transformer_replacement
    if int(train_cfg.max_steps) == 0 and not bool(
        getattr(train_cfg, "zero_shot_compile", False)
    ):
        raise ValueError(
            "max_steps=0 executes ZERO training steps. Set "
            "training.transformer_replacement.zero_shot_compile: true to "
            "run an explicit compile/calibrate-only conversion, or set "
            "max_steps > 0 to train. (Evidence-labeling guard, audit "
            "2026-08-16.)"
        )
    teacher, units = _load_hf_teacher_and_units(config, device=device, dtype=dtype)
    train_tokens, valid_tokens = _read_token_source(config)
    warm_started = _load_layerwise_warm_start(
        config,
        units,
        device=device,
    )
    for unit in units:
        unit._init_param_snapshot = {
            name: param.detach().cpu().clone()
            for name, param in unit.replacement.named_parameters()
        }
    trajectory_target = _uses_trajectory_target(config)
    capture = _MLPIOCapture(
        teacher,
        [unit.layer_index for unit in units],
        layers_attr=model_cfg.layers_attr,
        mlp_attr=model_cfg.target_module,
        capture_layer_hidden=trajectory_target,
    )
    if not warm_started and not str(getattr(train_cfg, "resume_checkpoint", "") or ""):
        _calibrate_hf_replacements(
            teacher,
            units,
            valid_tokens,
            capture,
            config,
            device=device,
            dtype=dtype,
        )
    forward_modules = _replacement_forward_modules(units, train_cfg, device)
    trainable_modules = nn.ModuleList(forward_modules)
    optimizer = create_optimizer(
        trainable_modules.parameters(), config.training.main.optimizer
    )
    scaler = _make_cuda_grad_scaler(train_cfg.use_amp, device)
    input_device = next(teacher.parameters()).device
    eval_batches = max(
        1, int(train_cfg.valid_samples) // max(1, int(train_cfg.batch_size))
    )

    start_step, history = _restore_layerwise_training(
        config,
        units,
        optimizer,
        scaler,
        device=device,
    )
    if history is None:
        initial_valid = _evaluate_hf_text(
            teacher,
            units,
            valid_tokens,
            capture,
            config,
            device=device,
            dtype=dtype,
            n_batches=eval_batches,
            forward_modules=forward_modules,
        )
        history = ReplacementTrainingHistory.start(initial_valid)
        _save_initial_best_replacement(config, units, history)

    pair_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    if bool(getattr(train_cfg, "teacher_pair_cache", False)):
        if isinstance(
            train_tokens, (StreamingTokenBatchSource, FrozenWindowBatchSource)
        ):
            raise NotImplementedError(
                "teacher_pair_cache requires an in-memory token pool; the "
                "restartable token source is consumed through its own batch policy"
            )
        if trajectory_target:
            raise NotImplementedError(
                "teacher_pair_cache does not support trajectory targets"
            )
        sequence_length = int(train_cfg.sequence_length)
        n_windows = int(train_tokens.numel()) // sequence_length
        max_windows = int(getattr(train_cfg, "teacher_pair_cache_max_windows", 256))
        n_windows = min(n_windows, max_windows)
        if n_windows < 2:
            raise ValueError(
                "teacher_pair_cache needs at least two aligned windows in "
                "the training pool"
            )
        logger.info(
            "teacher_pair_cache: capturing %d aligned windows once "
            "(replaces %d per-step teacher forwards)",
            n_windows,
            int(train_cfg.max_steps),
        )
        window_starts = [i * sequence_length for i in range(n_windows)]
        chunks: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {
            unit.layer_index: [] for unit in units
        }
        capture_batch = max(1, int(train_cfg.batch_size))
        with torch.no_grad():
            for chunk_start in range(0, n_windows, capture_batch):
                starts = window_starts[chunk_start : chunk_start + capture_batch]
                window_ids = torch.stack(
                    [train_tokens[start : start + sequence_length] for start in starts]
                ).to(device=input_device)
                capture.clear()
                chunk_pairs = _capture_hf_distillation_pairs(
                    teacher,
                    units,
                    capture,
                    window_ids,
                    trajectory=False,
                )
                for layer_index, (x, target) in chunk_pairs.items():
                    chunks[layer_index].append(
                        (x.detach().cpu(), target.detach().cpu())
                    )
        pair_cache = {
            layer_index: (
                torch.cat([x for x, _ in parts], dim=0),
                torch.cat([target for _, target in parts], dim=0),
            )
            for layer_index, parts in chunks.items()
        }
        cache_rng = torch.Generator().manual_seed(
            int(getattr(config.experiment, "seed", 0) or 0) + 977
        )

    optimizer_holder = {"optimizer": optimizer}

    def _run_training_steps(first_step: int, last_step: int) -> None:
        optimizer = optimizer_holder["optimizer"]
        for step in range(int(first_step), int(last_step) + 1):
            optimizer.zero_grad(set_to_none=True)
            if pair_cache is not None:
                any_layer = next(iter(pair_cache))
                pool_size = int(pair_cache[any_layer][0].shape[0])
                window_index = torch.randint(
                    0,
                    pool_size,
                    (int(train_cfg.batch_size),),
                    generator=cache_rng,
                )
                pairs = {
                    layer_index: (x[window_index], target[window_index])
                    for layer_index, (x, target) in pair_cache.items()
                }
            else:
                capture.clear()
                input_ids = _sample_token_batch(
                    train_tokens,
                    batch_size=train_cfg.batch_size,
                    sequence_length=train_cfg.sequence_length,
                    device=input_device,
                )
                pairs = _capture_hf_distillation_pairs(
                    teacher,
                    units,
                    capture,
                    input_ids,
                    trajectory=trajectory_target,
                )

            step_loss = torch.zeros((), device=device, dtype=torch.float32)
            for unit, module in zip(units, forward_modules):
                x, target = pairs[unit.layer_index]
                x = x.to(device=device, dtype=dtype)
                # fp32 targets preserve the trajectory correction, which can
                # round away entirely under bf16; the explicit .float() on the
                # module output gives autograd a cast node at the bf16/fp32
                # boundary so the fp32 loss backwards cleanly into bf16 params.
                target = target.to(device=device, dtype=torch.float32)
                module.train()
                with autocast("cuda", enabled=scaler.is_enabled()):
                    loss = _compute_distillation_loss(
                        module(x).float(),
                        target,
                        loss_name=train_cfg.loss,
                        cosine_weight=train_cfg.cosine_weight,
                    )
                step_loss = step_loss + loss.float() / max(len(units), 1)

            scaler.scale(step_loss).backward()
            _clip_replacement_grad_norm(
                trainable_modules.parameters(),
                max_grad_norm=getattr(train_cfg, "max_grad_norm", None),
                scaler=scaler,
                optimizer=optimizer,
            )
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            apply_sparse_topology_updates_after_scaled_step(
                trainable_modules,
                optimizer,
                scaler,
                scale_before=scale_before,
            )
            history.record_train_loss(
                transformer_distributed_mean(
                    float(step_loss.detach().item()),
                    device,
                )
            )

            if _should_run_periodic_step(
                step,
                train_cfg.eval_every,
                last_step,
            ):
                valid = _evaluate_hf_text(
                    teacher,
                    units,
                    valid_tokens,
                    capture,
                    config,
                    device=device,
                    dtype=dtype,
                    n_batches=eval_batches,
                    forward_modules=forward_modules,
                )
                _record_valid_and_maybe_save_best(
                    config,
                    units,
                    history,
                    step=step,
                    valid_loss=valid,
                )
                _save_token_stream_state(train_tokens, train_cfg.save_dir)
            if _should_run_periodic_step(step, train_cfg.log_every, 1):
                logger.info(
                    "hf_text dendritic distillation step=%s/%s train_loss=%.6f valid_loss=%.6f",
                    step,
                    train_cfg.max_steps,
                    history.train_losses[-1],
                    history.valid_losses[-1],
                )
            _maybe_save_layerwise_training(
                config,
                units,
                optimizer,
                scaler,
                history,
                step=step,
                device=device,
                token_source=train_tokens,
            )

    pruning_ladder_report: list[dict[str, Any]] | None = None
    try:
        _run_training_steps(start_step + 1, int(train_cfg.max_steps))
        _restore_best_replacement(config, units, device=device)
        ladder_rungs = validate_pruning_ladder_config(
            getattr(train_cfg, "pruning_ladder", None)
        )
        if ladder_rungs:
            if str(
                getattr(train_cfg, "distributed_mode", "none") or "none"
            ).lower() != "none" or bool(getattr(train_cfg, "data_parallel", False)):
                raise NotImplementedError(
                    "pruning_ladder requires single-process training: pruning "
                    "replaces Parameter objects, which invalidates DDP buckets "
                    "and DataParallel replicas"
                )
            if not bool(getattr(train_cfg, "restore_best_replacement", True)):
                raise ValueError(
                    "pruning_ladder requires restore_best_replacement: without "
                    "the best-checkpoint machinery a rung cannot restore its "
                    "within-rung best state"
                )

            def _ladder_evaluate() -> float:
                value = _evaluate_hf_text(
                    teacher,
                    units,
                    valid_tokens,
                    capture,
                    config,
                    device=device,
                    dtype=dtype,
                    n_batches=eval_batches,
                    forward_modules=forward_modules,
                )
                return float(value)

            def _ladder_rebuild_optimizer() -> None:
                optimizer_holder["optimizer"] = create_optimizer(
                    trainable_modules.parameters(),
                    config.training.main.optimizer,
                )

            pruning_ladder_report = run_layerwise_pruning_ladder_(
                ladder_rungs,
                units,
                history,
                start_step=int(train_cfg.max_steps),
                evaluate=_ladder_evaluate,
                run_steps=_run_training_steps,
                rebuild_optimizer=_ladder_rebuild_optimizer,
                save_best=lambda step, valid_loss: _save_best_replacement_checkpoint(
                    train_cfg.save_dir,
                    units,
                    step=step,
                    valid_loss=valid_loss,
                ),
                restore_best=lambda: _restore_best_replacement(
                    config, units, device=device
                ),
            )
        layer_metrics = _evaluate_hf_text_prediction_metrics(
            teacher,
            units,
            valid_tokens,
            capture,
            config,
            device=device,
            dtype=dtype,
            n_batches=eval_batches,
            forward_modules=forward_modules,
        )
    finally:
        capture.close()

    _save_token_stream_state(train_tokens, train_cfg.save_dir)

    steps_executed = len(history.train_losses)
    # Update diagnostics (audit 2026-08-16): "trained" requires the
    # checkpoint to demonstrably differ from init — BF16 params at small lr
    # round every update to zero while steps still "run".
    fraction_changed = None
    if steps_executed > 0:
        changed = total = 0
        for unit in units:
            snap = getattr(unit, "_init_param_snapshot", None)
            if snap is None:
                continue
            current = dict(unit.replacement.named_parameters())
            for name, init_val in snap.items():
                now = current.get(name)
                if now is None:
                    continue
                now_cpu = now.detach().cpu()
                if now_cpu.shape != init_val.shape:
                    # A pruning-ladder rung rebuilt this parameter at a new
                    # contact count; the topology change IS a change.
                    changed += init_val.numel()
                    total += init_val.numel()
                    continue
                changed += int((now_cpu != init_val).sum())
                total += init_val.numel()
        if total:
            fraction_changed = changed / total
            logger.info(
                "update diagnostics: %d/%d params changed (%.6f%%)",
                changed,
                total,
                100.0 * fraction_changed,
            )
    mode = "zero_shot_compile"
    if steps_executed > 0:
        mode = (
            "trained"
            if fraction_changed is None or fraction_changed > 1e-4
            else "trained_no_update"
        )
        if mode == "trained_no_update":
            logger.warning(
                "Ran %d steps but parameters effectively unchanged "
                "(%.6f%%) — check lr and replacement_dtype.",
                steps_executed,
                100.0 * (fraction_changed or 0.0),
            )
    if pruning_ladder_report is not None:
        ladder_path = Path(train_cfg.save_dir) / "pruning_ladder_report.json"
        ladder_path.parent.mkdir(parents=True, exist_ok=True)
        ladder_path.write_text(
            json.dumps(
                {
                    "schema": "dendritic_replacement_pruning_ladder_report/v1",
                    "status": "measured",
                    "rungs": pruning_ladder_report,
                },
                indent=2,
                default=str,
            )
        )
        logger.info(
            "pruning ladder complete: %d rung(s), report at %s",
            len(pruning_ladder_report),
            ladder_path,
        )
    result = TransformerReplacementTrainingResult(
        **history.as_result_kwargs(),
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
        layer_metrics=layer_metrics,
        execution_mode=mode,
        steps_requested=int(train_cfg.max_steps),
        steps_executed=steps_executed,
    )
    _save_result(
        result,
        units,
        save_replacements=train_cfg.save_replacements,
        topology_encoding=str(train_cfg.replacement_checkpoint_encoding),
        freeze_sparse_topology_on_export=bool(
            getattr(train_cfg, "freeze_sparse_topology_on_export", False)
        ),
        describe_fixed_topology=pruning_ladder_report is not None,
        stochastic_topology_freeze_policy=str(
            getattr(train_cfg, "stochastic_topology_freeze_policy", "reject")
        ),
        stochastic_topology_freeze_seed=int(
            getattr(train_cfg, "stochastic_topology_freeze_seed", 0)
        ),
        ragged_topology_format=str(getattr(train_cfg, "ragged_topology_format", "csr")),
    )
    return result


def _validate_distributed_joint_validation_contract(train_cfg: object) -> None:
    """Reject joint objectives that cannot be reconstructed from rank means."""

    if transformer_process_world_size() == 1:
        return
    unsupported = []
    if float(getattr(train_cfg, "sequence_risk_weight", 0.0)) != 0.0:
        unsupported.append("sequence_risk_weight")
    if float(getattr(train_cfg, "validation_tail_weight", 0.0)) != 0.0:
        unsupported.append("validation_tail_weight")
    if float(getattr(train_cfg, "teacher_topk_margin_weight", 0.0)) != 0.0:
        unsupported.append("teacher_topk_margin_weight")
    hidden_weight = float(getattr(train_cfg, "hidden_loss_weight", 0.0))
    hidden_type = str(getattr(train_cfg, "hidden_loss_type", "mse")).lower()
    if hidden_weight != 0.0 and hidden_type in {
        "relative_mse",
        "relative_mse_cosine",
    }:
        unsupported.append(f"hidden_loss_type={hidden_type}")
    if unsupported:
        raise NotImplementedError(
            "exact DDP joint validation requires decomposable mean objectives; "
            "per-sequence/global-denominator aggregation is not implemented for "
            + ", ".join(unsupported)
        )


def _resolve_joint_validation_batches(
    train_cfg: object,
    valid_tokens: torch.Tensor,
) -> int:
    """Resolve joint-validation batches without changing legacy semantics.

    ``valid_samples`` historically meant a per-rank count.  Formal distributed
    comparisons can additionally bind the exact global window count.  Opting
    into that contract requires the legacy field to equal the corresponding
    per-rank share and can prohibit the modulo reuse performed by the
    deterministic validation sampler.
    """

    batch_size = int(train_cfg.batch_size)
    valid_samples = int(train_cfg.valid_samples)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if valid_samples < 1:
        raise ValueError("valid_samples must be positive")

    global_samples_raw = getattr(train_cfg, "validation_global_samples", None)
    if global_samples_raw is None:
        return max(1, valid_samples // batch_size)
    if isinstance(global_samples_raw, bool):
        raise TypeError("validation_global_samples must be a positive integer")
    global_samples = int(global_samples_raw)
    if global_samples < 1:
        raise ValueError("validation_global_samples must be positive")

    world_size = transformer_process_world_size()
    global_batch_size = batch_size * world_size
    if global_samples % global_batch_size:
        raise ValueError(
            "validation_global_samples must be divisible by batch_size * world_size"
        )
    expected_per_rank = global_samples // world_size
    if valid_samples != expected_per_rank:
        raise ValueError(
            "valid_samples must equal validation_global_samples / world_size "
            f"({expected_per_rank}), received {valid_samples}"
        )

    allow_wrap = getattr(train_cfg, "validation_allow_window_wrap", True)
    if not isinstance(allow_wrap, bool):
        raise TypeError("validation_allow_window_wrap must be boolean")
    if not allow_wrap:
        if valid_tokens.ndim == 2:
            available_windows = int(valid_tokens.shape[0])
        elif valid_tokens.ndim == 1:
            sequence_length = int(train_cfg.sequence_length)
            if sequence_length < 1:
                raise ValueError("sequence_length must be positive")
            available_windows = int(valid_tokens.numel()) // sequence_length
        else:
            raise ValueError("Validation token source must be rank one or rank two")
        if global_samples > available_windows:
            raise ValueError(
                "non-wrapping joint validation requested "
                f"{global_samples} windows but only {available_windows} are available"
            )
    return global_samples // global_batch_size


def _joint_checkpoint_selection(
    metrics: dict[str, float],
    train_cfg: object,
) -> tuple[float, str]:
    """Return the prospectively configured scalar and its explicit identity."""

    metric = (
        str(getattr(train_cfg, "checkpoint_selection_metric", "objective"))
        .strip()
        .lower()
    )
    if metric == "objective":
        return float(metrics["selection"]), "objective"
    if metric == "lm":
        return float(metrics["lm"]), "lm"
    if metric == "kl":
        if float(getattr(train_cfg, "kl_loss_weight", 0.0)) <= 0:
            raise ValueError("KL checkpoint selection requires kl_loss_weight > 0")
        return float(metrics["kl"]), "kl"
    raise ValueError("checkpoint_selection_metric must be 'objective', 'lm', or 'kl'")


def _resolve_ddp_state_digest_steps(train_cfg: object) -> tuple[int, ...]:
    """Resolve periodic-prefix and explicit DDP integrity gates exactly."""

    maximum = int(train_cfg.max_steps)
    every_raw = getattr(train_cfg, "ddp_state_digest_every", 0)
    qualification_raw = getattr(
        train_cfg,
        "ddp_state_digest_qualification_steps",
        0,
    )
    explicit_raw = list(getattr(train_cfg, "ddp_state_digest_steps", []) or [])
    for value, label in (
        (every_raw, "ddp_state_digest_every"),
        (qualification_raw, "ddp_state_digest_qualification_steps"),
    ):
        if isinstance(value, bool):
            raise ValueError(f"{label} must be a non-negative integer")
    every = int(every_raw)
    qualification = int(qualification_raw)
    if every < 0:
        raise ValueError("ddp_state_digest_every must be a non-negative integer")
    if not 0 <= qualification <= maximum:
        raise ValueError(
            "ddp_state_digest_qualification_steps must lie between zero and max_steps"
        )
    if qualification and not every:
        raise ValueError(
            "ddp_state_digest_qualification_steps requires ddp_state_digest_every"
        )
    if any(
        isinstance(step, bool) or not isinstance(step, int) for step in explicit_raw
    ):
        raise TypeError("ddp_state_digest_steps must contain integers")
    explicit = [int(step) for step in explicit_raw]
    if explicit != sorted(set(explicit)):
        raise ValueError("ddp_state_digest_steps must be sorted and unique")
    if any(step <= 0 or step > maximum for step in explicit):
        raise ValueError("ddp_state_digest_steps must lie in [1, max_steps]")

    resolved = set(explicit)
    if every:
        limit = qualification if qualification else maximum
        resolved.update(range(every, limit + 1, every))
        resolved.add(limit)
    return tuple(sorted(resolved))


def _configure_transformer_execution_precision(
    experiment_cfg: object,
) -> dict[str, Any]:
    """Apply and receipt the requested transformer floating-point policy."""

    requested_precision = str(
        getattr(experiment_cfg, "float32_matmul_precision", "highest")
    ).lower()
    if requested_precision not in {"highest", "high", "medium"}:
        raise ValueError(
            "experiment.float32_matmul_precision must be highest, high, or medium"
        )
    requested_tf32 = bool(getattr(experiment_cfg, "allow_tf32", False))
    strict_deterministic = bool(getattr(experiment_cfg, "strict_deterministic", False))
    effective_tf32 = requested_tf32 and not strict_deterministic

    torch.backends.cuda.matmul.allow_tf32 = effective_tf32
    torch.backends.cudnn.allow_tf32 = effective_tf32
    torch.set_float32_matmul_precision(requested_precision)
    observed_precision = str(torch.get_float32_matmul_precision()).lower()
    observed_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    observed_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    if (
        observed_precision != requested_precision
        or observed_matmul_tf32 != effective_tf32
        or observed_cudnn_tf32 != effective_tf32
    ):
        raise RuntimeError("PyTorch did not apply the requested precision policy")
    return {
        "schema": "dendritic_transformer_execution_precision/v1",
        "requested": {
            "allow_tf32": requested_tf32,
            "float32_matmul_precision": requested_precision,
            "strict_deterministic": strict_deterministic,
        },
        "effective": {
            "allow_tf32": effective_tf32,
            "cuda_matmul_allow_tf32": observed_matmul_tf32,
            "cudnn_allow_tf32": observed_cudnn_tf32,
            "float32_matmul_precision": observed_precision,
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve_joint_checkpoint_steps(train_cfg: object) -> tuple[int, ...]:
    """Resolve one immutable joint checkpoint schedule."""

    max_steps = int(train_cfg.max_steps)
    raw_steps = list(getattr(train_cfg, "joint_checkpoint_steps", []) or [])
    checkpoint_every = int(getattr(train_cfg, "checkpoint_every", 0) or 0)
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if raw_steps and checkpoint_every:
        raise ValueError(
            "joint_checkpoint_steps and checkpoint_every are mutually exclusive"
        )
    if raw_steps:
        if any(isinstance(step, bool) for step in raw_steps):
            raise TypeError("joint_checkpoint_steps must contain integers")
        steps = tuple(int(step) for step in raw_steps)
        if list(steps) != sorted(set(steps)):
            raise ValueError("joint_checkpoint_steps must be sorted and unique")
        if steps[0] != 0 or steps[-1] != max_steps:
            raise ValueError(
                "joint_checkpoint_steps must begin at zero and end at max_steps"
            )
        if any(step < 0 or step > max_steps for step in steps):
            raise ValueError("joint_checkpoint_steps must lie within training")
        return steps
    if not checkpoint_every:
        return ()
    periodic = list(range(checkpoint_every, max_steps + 1, checkpoint_every))
    if not periodic or periodic[-1] != max_steps:
        periodic.append(max_steps)
    return (0, *periodic)


def _joint_checkpoint_path(save_dir: str, step: int) -> str:
    return os.path.join(
        save_dir,
        "joint_checkpoints",
        f"step_{int(step):06d}",
        "training_checkpoint.pt",
    )


def _joint_restart_contract(
    train_cfg: object,
    *,
    selection_metric: str,
    eval_batches: int,
) -> dict[str, Any]:
    """Bind the training semantics that must not change across a restart."""

    execution_contract = getattr(
        train_cfg,
        "joint_restart_execution_contract",
        {},
    )
    if not isinstance(execution_contract, Mapping):
        raise TypeError("joint_restart_execution_contract must be a mapping")
    canonical_execution = json.dumps(
        dict(execution_contract),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "schema": "dendritic_joint_restart_contract/v1",
        "max_steps": int(train_cfg.max_steps),
        "eval_every": int(train_cfg.eval_every),
        "log_every": int(train_cfg.log_every),
        "joint_checkpoint_steps": [
            int(value)
            for value in (getattr(train_cfg, "joint_checkpoint_steps", []) or [])
        ],
        "ddp_state_digest_every": int(getattr(train_cfg, "ddp_state_digest_every", 0)),
        "ddp_state_digest_qualification_steps": int(
            getattr(train_cfg, "ddp_state_digest_qualification_steps", 0)
        ),
        "ddp_state_digest_steps": [
            int(value)
            for value in (getattr(train_cfg, "ddp_state_digest_steps", []) or [])
        ],
        "train_target": str(train_cfg.train_target),
        **(
            {
                "joint_recovery_policy": {
                    "layer_lr_multipliers": dict(train_cfg.joint_layer_lr_multipliers),
                    "proximal_weight": float(train_cfg.joint_proximal_weight),
                    "proximal_layers": list(train_cfg.joint_proximal_layers),
                    "proximal_epsilon": float(train_cfg.joint_proximal_epsilon),
                    "restart_supported": False,
                }
            }
            if getattr(train_cfg, "joint_layer_lr_multipliers", {})
            or float(getattr(train_cfg, "joint_proximal_weight", 0.0)) > 0
            else {}
        ),
        "joint_global_teacher_role": str(
            getattr(train_cfg, "joint_global_teacher_role", "student_base")
        ),
        "joint_global_teacher_model_kwargs": dict(
            getattr(train_cfg, "joint_global_teacher_model_kwargs", {}) or {}
        ),
        "batch_size_per_rank": int(train_cfg.batch_size),
        "sequence_length": int(train_cfg.sequence_length),
        "valid_samples_per_rank": int(train_cfg.valid_samples),
        "validation_global_samples": getattr(
            train_cfg, "validation_global_samples", None
        ),
        "validation_batches_per_rank": int(eval_batches),
        "validation_allow_window_wrap": bool(
            getattr(train_cfg, "validation_allow_window_wrap", True)
        ),
        "checkpoint_selection_metric": str(selection_metric),
        "lm_loss_weight": float(train_cfg.lm_loss_weight),
        "kl_loss_weight": float(train_cfg.kl_loss_weight),
        "kl_temperature": float(train_cfg.kl_temperature),
        "hidden_loss_weight": float(train_cfg.hidden_loss_weight),
        "hidden_loss_layers": [int(value) for value in train_cfg.hidden_loss_layers],
        "hidden_loss_type": str(train_cfg.hidden_loss_type),
        "hidden_loss_layer_weights": [
            float(value) for value in train_cfg.hidden_loss_layer_weights
        ],
        "hidden_loss_epsilon": float(train_cfg.hidden_loss_epsilon),
        "sequence_risk_weight": float(train_cfg.sequence_risk_weight),
        "sequence_risk_fraction": float(train_cfg.sequence_risk_fraction),
        "validation_tail_weight": float(train_cfg.validation_tail_weight),
        "validation_tail_fraction": float(train_cfg.validation_tail_fraction),
        "teacher_topk_margin_weight": float(train_cfg.teacher_topk_margin_weight),
        "teacher_topk_margin_k": int(train_cfg.teacher_topk_margin_k),
        "teacher_topk_margin_epsilon": float(train_cfg.teacher_topk_margin_epsilon),
        "dtype": str(train_cfg.dtype),
        "replacement_dtype": str(train_cfg.replacement_dtype),
        "use_amp": bool(train_cfg.use_amp),
        "gradient_checkpointing": bool(train_cfg.gradient_checkpointing),
        "max_grad_norm": (
            None
            if getattr(train_cfg, "max_grad_norm", None) is None
            else float(train_cfg.max_grad_norm)
        ),
        "seed": (
            None if getattr(train_cfg, "seed", None) is None else int(train_cfg.seed)
        ),
        "execution_contract": dict(execution_contract),
        "execution_contract_sha256": hashlib.sha256(
            canonical_execution.encode("utf-8")
        ).hexdigest(),
    }


def _evaluate_joint_distillation_on_tokens(
    student: nn.Module,
    teacher: nn.Module | None,
    tokens: torch.Tensor,
    config: Config,
    *,
    hidden_layers: Sequence[int],
    n_batches: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the same LM/KL/hidden objective used by joint training.

    Historical joint runs selected checkpoints using LM loss even when the
    configured objective was dominated by hidden-state or KL distillation.
    This evaluator keeps validation and training objectives aligned, which is
    required for clean span-exit distillation and best-rung restoration.
    """

    train_cfg = config.training.transformer_replacement
    _validate_distributed_joint_validation_contract(train_cfg)
    output_hidden = float(train_cfg.hidden_loss_weight) > 0
    input_device = next(student.parameters()).device
    amp_enabled = bool(train_cfg.use_amp) and device.type == "cuda"
    component_names = (
        "total",
        "lm",
        "lm_mean",
        "lm_cvar",
        "kl",
        "teacher_topk_margin",
        "hidden",
    )
    values = {name: [] for name in component_names}
    student.eval()
    if teacher is not None:
        teacher.eval()
    with torch.no_grad():
        for batch_index in range(max(1, int(n_batches))):
            input_ids = _deterministic_validation_token_batch(
                tokens,
                batch_index=batch_index,
                batch_size=int(train_cfg.batch_size),
                sequence_length=int(train_cfg.sequence_length),
                device=input_device,
            )
            teacher_output = None
            if teacher is not None:
                with autocast("cuda", enabled=amp_enabled):
                    teacher_output = teacher(
                        input_ids=input_ids,
                        labels=input_ids,
                        output_hidden_states=output_hidden,
                    )
            with autocast("cuda", enabled=amp_enabled):
                student_output = student(
                    input_ids=input_ids,
                    labels=input_ids,
                    output_hidden_states=output_hidden,
                )
                _loss, components = _joint_lm_distillation_loss(
                    student_output,
                    teacher_output,
                    hidden_layers=hidden_layers,
                    lm_loss_weight=float(train_cfg.lm_loss_weight),
                    kl_loss_weight=float(train_cfg.kl_loss_weight),
                    kl_temperature=float(train_cfg.kl_temperature),
                    hidden_loss_weight=float(train_cfg.hidden_loss_weight),
                    hidden_loss_type=str(train_cfg.hidden_loss_type),
                    hidden_loss_layer_weights=list(train_cfg.hidden_loss_layer_weights),
                    hidden_loss_epsilon=float(train_cfg.hidden_loss_epsilon),
                    labels=input_ids,
                    sequence_risk_weight=float(train_cfg.sequence_risk_weight),
                    sequence_risk_fraction=float(train_cfg.sequence_risk_fraction),
                    teacher_topk_margin_weight=float(
                        train_cfg.teacher_topk_margin_weight
                    ),
                    teacher_topk_margin_k=int(train_cfg.teacher_topk_margin_k),
                    teacher_topk_margin_epsilon=float(
                        train_cfg.teacher_topk_margin_epsilon
                    ),
                )
            local_row = [float(components[name]) for name in component_names]
            if transformer_process_world_size() > 1:
                assert_transformer_distributed_finite_row(
                    local_row,
                    device,
                    stage=f"joint_validation_batch_{batch_index}",
                )
            rank_rows = transformer_distributed_gather_row(local_row, device)
            if any(len(row) != len(component_names) for row in rank_rows):
                raise RuntimeError("distributed joint validation row width differs")
            for component_index, name in enumerate(component_names):
                values[name].append(
                    sum(row[component_index] for row in rank_rows) / len(rank_rows)
                )

    metrics: dict[str, float] = {}
    for name, samples in values.items():
        tensor = torch.tensor(samples, dtype=torch.float64)
        metrics[name] = float(tensor.mean().item())
        metrics[f"{name}_median"] = float(torch.quantile(tensor, 0.5).item())
        metrics[f"{name}_p90"] = float(torch.quantile(tensor, 0.9).item())
        metrics[f"{name}_p95"] = float(torch.quantile(tensor, 0.95).item())
        metrics[f"{name}_max"] = float(tensor.max().item())
    tail_fraction = float(train_cfg.validation_tail_fraction)
    total_tensor = torch.tensor(values["total"], dtype=torch.float64)
    total_cvar = float(_upper_tail_mean(total_tensor, tail_fraction).item())
    tail_weight = float(train_cfg.validation_tail_weight)
    if not 0.0 <= tail_weight <= 1.0:
        raise ValueError("validation_tail_weight must lie in [0, 1]")
    metrics["total_cvar_across_batches"] = total_cvar
    metrics["selection"] = (1.0 - tail_weight) * metrics[
        "total"
    ] + tail_weight * total_cvar
    return metrics


def _joint_global_teacher_load_context(
    config: Config,
    model_source_session: TransformerModelSourceSession | None,
) -> tuple[Config, TransformerModelSourceSession | None, str]:
    """Resolve global supervision independently from the student base."""

    train_cfg = config.training.transformer_replacement
    role = (
        str(getattr(train_cfg, "joint_global_teacher_role", "student_base"))
        .strip()
        .lower()
    )
    if role == "student_base":
        return config, model_source_session, role
    if role != "model_name_pretrained":
        raise ValueError(
            "joint_global_teacher_role must be 'student_base' or "
            "'model_name_pretrained'"
        )
    teacher_config = copy.deepcopy(config)
    teacher_model_cfg = teacher_config.model.transformer_replacement
    teacher_model_cfg.model_source = {}
    teacher_model_cfg.model_kwargs = dict(
        getattr(train_cfg, "joint_global_teacher_model_kwargs", {}) or {}
    )
    if not str(teacher_model_cfg.model_name).strip():
        raise ValueError("model_name_pretrained global teacher has no model_name")
    return teacher_config, None, role


def _train_joint_hf_text_source(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformerReplacementTrainingResult:
    """Train all patched dendritic MLPs through the full LM objective."""
    train_cfg = config.training.transformer_replacement
    model_cfg = config.model.transformer_replacement
    validate_joint_recovery_policy_config(train_cfg)

    train_tokens, valid_tokens = _read_token_source(config)
    model_source_declaration = dict(getattr(model_cfg, "model_source", {}) or {})
    model_source_session = (
        TransformerModelSourceSession(model_source_declaration)
        if model_source_declaration
        else None
    )
    teacher_needed = bool(
        float(train_cfg.kl_loss_weight) > 0
        or float(train_cfg.hidden_loss_weight) > 0
        or float(train_cfg.teacher_topk_margin_weight) > 0
    )
    teacher_config, teacher_model_source_session, global_teacher_role = (
        _joint_global_teacher_load_context(config, model_source_session)
    )
    teacher = (
        _load_hf_causal_lm_for_joint_training(
            teacher_config,
            device=device,
            dtype=dtype,
            initialization="pretrained",
            model_source_session=teacher_model_source_session,
        )
        if teacher_needed
        else None
    )
    if teacher is not None:
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    student = _load_hf_causal_lm_for_joint_training(
        config,
        device=device,
        dtype=dtype,
        initialization=str(getattr(train_cfg, "student_initialization", "pretrained")),
        model_source_session=model_source_session,
    )
    if model_source_session is not None:
        model_source_receipt = model_source_session.finalize()
        student._dendritic_model_source_receipt = model_source_receipt
        if teacher is not None and global_teacher_role == "student_base":
            teacher._dendritic_model_source_receipt = model_source_receipt
        if is_transformer_main_process():
            source_receipt_path = Path(str(train_cfg.save_dir)) / (
                "transformer_model_source_receipt.json"
            )
            source_receipt_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = source_receipt_path.with_name(
                f".{source_receipt_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(
                json.dumps(model_source_receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, source_receipt_path)
    replacement_layers = resolve_configured_replacement_layers(config)
    student_layers = resolve_transformer_layers(
        student,
        layers_attr=model_cfg.layers_attr,
    )
    _apply_pre_patched_replacements(
        student,
        student_layers,
        config,
        layer_indices=replacement_layers,
        device=device,
    )
    pre_patched_collapsed_records = _apply_pre_patched_collapsed_spans(
        student,
        config,
        layer_indices=replacement_layers,
        device=device,
    )
    records = apply_transformer_replacement_config(
        student,
        model_cfg,
        config.model.core,
    )
    if not records:
        raise ValueError("No transformer replacement records were created")
    warm_start_dir = str(getattr(train_cfg, "warm_start_dir", "") or "")
    resume_checkpoint = str(getattr(train_cfg, "resume_checkpoint", "") or "")
    if warm_start_dir and resume_checkpoint:
        raise ValueError(
            "warm_start_dir and resume_checkpoint are mutually exclusive: "
            "warm starts begin a new recovery while resume restores one"
        )
    if warm_start_dir:
        logger.info(
            "Loading dendritic replacement warm-start from %s",
            warm_start_dir,
        )
        _load_replacement_record_checkpoints(
            records,
            warm_start_dir,
            device=device,
        )

    offloaded_span_ffns = _offload_collapsed_span_original_mlps_(
        [*pre_patched_collapsed_records, *records]
    )
    if offloaded_span_ffns:
        logger.info(
            "Offloaded %d detached dense collapsed-span FFNs to CPU",
            offloaded_span_ffns,
        )

    train_target = str(getattr(train_cfg, "train_target", "replacement_only"))
    trainable_params = _select_joint_trainable_parameters(
        student,
        records,
        train_target,
    )
    train_full_model = train_target.lower() in {"full_model", "all", "student"}
    use_gradient_checkpointing = bool(
        getattr(train_cfg, "gradient_checkpointing", False)
    )
    if use_gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        # Reentrant checkpointing (the transformers default) requires some
        # checkpointed-block INPUT to carry requires_grad; with
        # replacement-only training every non-replacement parameter
        # (embeddings included) is frozen, so reentrant checkpoint outputs
        # detach and loss.backward() raises. Non-reentrant checkpointing plus
        # grads-on-embedding-outputs keeps replacement gradients flowing.
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
    if use_gradient_checkpointing or train_full_model:
        student.train()
    else:
        student.eval()
    for record in records:
        record.replacement.train()

    for record in pre_patched_collapsed_records:
        probe_input_ids = _sample_token_batch(
            valid_tokens,
            batch_size=train_cfg.batch_size,
            sequence_length=train_cfg.sequence_length,
            device=device,
        )
        contract = validate_collapsed_span_additive_contract(
            student,
            record,
            probe_input_ids,
            layers_attr=model_cfg.layers_attr,
        )
        cell = getattr(record.replacement, "span_cell", record.replacement)
        diagnostics = dict(getattr(cell, "teacher_topology_diagnostics", {}) or {})
        diagnostics["collapsed_span_additive_contract"] = contract
        cell.teacher_topology_diagnostics = diagnostics
        cell.teacher_topk_diagnostics = diagnostics

    _calibrate_joint_replacements_if_fresh(
        student,
        records,
        valid_tokens,
        config,
        device=device,
        teacher=teacher,
    )
    recovery_policy = build_joint_recovery_policy(
        records, train_cfg, config.training.main.optimizer
    )
    recovery_policy_history = []
    if recovery_policy is not None:
        trainable_params = recovery_policy.trainable_parameters
        recovery_policy.write_report(train_cfg.save_dir, recovery_policy_history)
    student_forward = wrap_transformer_ddp(student, train_cfg, device)

    ddp_state_digest_steps = _resolve_ddp_state_digest_steps(train_cfg)
    ddp_state_digest_every = int(getattr(train_cfg, "ddp_state_digest_every", 0))
    ddp_state_digest_qualification_steps = int(
        getattr(train_cfg, "ddp_state_digest_qualification_steps", 0)
    )
    ddp_state_digest_explicit_steps = tuple(
        int(value) for value in (getattr(train_cfg, "ddp_state_digest_steps", []) or [])
    )
    ddp_state_digest_enabled = bool(
        ddp_state_digest_every or ddp_state_digest_explicit_steps
    )
    if ddp_state_digest_enabled and transformer_process_world_size() == 1:
        raise ValueError("DDP state digests require a multi-rank DDP process group")
    ddp_state_digest_step_set = set(ddp_state_digest_steps)
    ddp_state_digest_records: list[dict[str, Any]] = []
    ddp_local_finite_loss_checks = 0
    ddp_distributed_finite_loss_checks = 0

    def _record_ddp_state_digest(stage: str) -> None:
        # The explicit pre/post synchronization makes the cryptographic gate's
        # wall time separable from optimizer-step compute in systems probes.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        digest_started = time.perf_counter()
        record = assert_transformer_distributed_module_state_equal(
            [replacement_record.replacement for replacement_record in records],
            stage=stage,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record["qualification_wall_time_seconds"] = time.perf_counter() - digest_started
        ddp_state_digest_records.append(record)

    if ddp_state_digest_enabled and not resume_checkpoint:
        _record_ddp_state_digest("post_ddp_initial_sync")

    optimizer = create_optimizer(
        (
            trainable_params
            if recovery_policy is None
            else recovery_policy.optimizer_groups
        ),
        config.training.main.optimizer,
    )
    scaler = _make_cuda_grad_scaler(train_cfg.use_amp, device)

    eval_batches = _resolve_joint_validation_batches(train_cfg, valid_tokens)
    hidden_layers = _resolve_joint_hidden_layers(config)
    units = _records_to_units(records)
    restore_joint_best = str(train_target).lower() in {
        "replacement_only",
        "replacements_only",
        "dendritic_only",
    }
    checkpoint_steps = _resolve_joint_checkpoint_steps(train_cfg)
    if (checkpoint_steps or resume_checkpoint) and not restore_joint_best:
        raise NotImplementedError(
            "joint restart checkpoints currently support replacement-only training"
        )
    if (checkpoint_steps or resume_checkpoint) and not bool(
        getattr(train_cfg, "restore_best_replacement", True)
    ):
        raise ValueError(
            "joint restart checkpoints require restore_best_replacement=true"
        )
    configured_selection_metric = (
        str(getattr(train_cfg, "checkpoint_selection_metric", "objective"))
        .strip()
        .lower()
    )
    if configured_selection_metric not in {"objective", "lm", "kl"}:
        raise ValueError("checkpoint_selection_metric must be 'objective', 'lm', or 'kl'")
    if configured_selection_metric == "kl" and float(train_cfg.kl_loss_weight) <= 0:
        raise ValueError("KL checkpoint selection requires kl_loss_weight > 0")
    restart_contract = _joint_restart_contract(
        train_cfg,
        selection_metric=configured_selection_metric,
        eval_batches=eval_batches,
    )
    start_step = 0
    if resume_checkpoint:
        start_step, history, validation_metrics_history = (
            _load_joint_training_checkpoint(
                resume_checkpoint,
                units,
                optimizer,
                scaler,
                train_tokens,
                restart_contract,
                device=device,
                save_dir=train_cfg.save_dir,
                ddp_integrity_history=ddp_state_digest_records,
            )
        )
        if start_step not in checkpoint_steps:
            raise ValueError(
                "resume checkpoint step is absent from joint_checkpoint_steps"
            )
        if start_step > int(train_cfg.max_steps):
            raise ValueError("resume checkpoint cannot exceed max_steps")
        selection_metric = configured_selection_metric
        if ddp_state_digest_enabled:
            _record_ddp_state_digest(f"post_joint_checkpoint_restore_step_{start_step}")
    else:
        initial_metrics = _evaluate_joint_distillation_on_tokens(
            student_forward,
            teacher,
            valid_tokens,
            config,
            hidden_layers=hidden_layers,
            n_batches=eval_batches,
            device=device,
        )
        initial_valid, selection_metric = _joint_checkpoint_selection(
            initial_metrics,
            train_cfg,
        )
        validation_metrics_history = [
            {
                "step": 0,
                "checkpoint_selection_value": initial_valid,
                **{name: float(value) for name, value in initial_metrics.items()},
            }
        ]
        history = ReplacementTrainingHistory.start(initial_valid)
        if restore_joint_best:
            _save_initial_best_replacement(config, units, history)
        if 0 in checkpoint_steps:
            _save_joint_training_checkpoint(
                _joint_checkpoint_path(train_cfg.save_dir, 0),
                units,
                optimizer,
                scaler,
                history,
                validation_metrics_history,
                train_tokens,
                restart_contract,
                step=0,
                device=device,
                save_dir=train_cfg.save_dir,
                ddp_integrity_history=ddp_state_digest_records,
            )
    input_device = next(student.parameters()).device

    for step in range(start_step + 1, int(train_cfg.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        input_ids = _sample_token_batch(
            train_tokens,
            batch_size=train_cfg.batch_size,
            sequence_length=train_cfg.sequence_length,
            device=input_device,
        )
        output_hidden = float(train_cfg.hidden_loss_weight) > 0
        teacher_output = None
        if teacher is not None:
            with torch.no_grad(), autocast("cuda", enabled=scaler.is_enabled()):
                teacher_output = teacher(
                    input_ids=input_ids,
                    labels=input_ids,
                    output_hidden_states=output_hidden,
                )

        if use_gradient_checkpointing or train_full_model:
            student.train()
        else:
            student.eval()
        for record in records:
            record.replacement.train()
        with autocast("cuda", enabled=scaler.is_enabled()):
            student_output = student_forward(
                input_ids=input_ids,
                labels=input_ids,
                output_hidden_states=output_hidden,
            )
            loss, components = _joint_lm_distillation_loss(
                student_output,
                teacher_output,
                hidden_layers=hidden_layers,
                lm_loss_weight=float(train_cfg.lm_loss_weight),
                kl_loss_weight=float(train_cfg.kl_loss_weight),
                kl_temperature=float(train_cfg.kl_temperature),
                hidden_loss_weight=float(train_cfg.hidden_loss_weight),
                hidden_loss_type=str(train_cfg.hidden_loss_type),
                hidden_loss_layer_weights=list(train_cfg.hidden_loss_layer_weights),
                hidden_loss_epsilon=float(train_cfg.hidden_loss_epsilon),
                labels=input_ids,
                sequence_risk_weight=float(train_cfg.sequence_risk_weight),
                sequence_risk_fraction=float(train_cfg.sequence_risk_fraction),
                teacher_topk_margin_weight=float(train_cfg.teacher_topk_margin_weight),
                teacher_topk_margin_k=int(train_cfg.teacher_topk_margin_k),
                teacher_topk_margin_epsilon=float(
                    train_cfg.teacher_topk_margin_epsilon
                ),
            )

        if recovery_policy is not None:
            loss, policy_row = recovery_policy.training_loss(
                loss, components, step=step
            )
            recovery_policy_history.append(policy_row)
            components["total"] = float(policy_row["training_total"])
            components["proximal_penalty"] = float(policy_row["proximal_penalty"])

        if ddp_state_digest_enabled and step in ddp_state_digest_step_set:
            # _joint_lm_distillation_loss already materializes every component,
            # including total, as a Python float for the historical logger.
            # Reuse those values so the digest-free phase adds no new GPU sync.
            finite_values = list(components.values())
            if not all(math.isfinite(float(value)) for value in finite_values):
                raise FloatingPointError(
                    f"non-finite local scalar at joint_train_step_{step}"
                )
            ddp_local_finite_loss_checks += 1
            assert_transformer_distributed_finite_row(
                finite_values,
                device,
                stage=f"joint_train_step_{step}",
            )
            ddp_distributed_finite_loss_checks += 1

        scaler.scale(loss).backward()
        _clip_replacement_grad_norm(
            trainable_params,
            max_grad_norm=getattr(train_cfg, "max_grad_norm", None),
            scaler=scaler,
            optimizer=optimizer,
        )
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        apply_sparse_topology_updates_after_scaled_step(
            student,
            optimizer,
            scaler,
            scale_before=scale_before,
        )
        if ddp_state_digest_enabled and step in ddp_state_digest_step_set:
            _record_ddp_state_digest(f"post_optimizer_and_topology_step_{step}")
        history.record_train_loss(
            transformer_distributed_mean(
                float(loss.detach().float().item()),
                device,
            )
        )

        if _should_run_periodic_step(step, train_cfg.eval_every, train_cfg.max_steps):
            valid_metrics = _evaluate_joint_distillation_on_tokens(
                student_forward,
                teacher,
                valid_tokens,
                config,
                hidden_layers=hidden_layers,
                n_batches=eval_batches,
                device=device,
            )
            valid, observed_selection_metric = _joint_checkpoint_selection(
                valid_metrics,
                train_cfg,
            )
            if observed_selection_metric != selection_metric:
                raise RuntimeError(
                    "checkpoint selection metric changed during training"
                )
            validation_metrics_history.append(
                {
                    "step": int(step),
                    "checkpoint_selection_value": valid,
                    **{name: float(value) for name, value in valid_metrics.items()},
                }
            )
            if restore_joint_best:
                _record_valid_and_maybe_save_best(
                    config,
                    units,
                    history,
                    step=step,
                    valid_loss=valid,
                )
            else:
                history.record_valid_loss(step, valid)
            _save_token_stream_state(train_tokens, train_cfg.save_dir)
            if recovery_policy is not None:
                recovery_policy.write_report(
                    train_cfg.save_dir, recovery_policy_history
                )
            for record in records:
                record.replacement.train()

        if step in checkpoint_steps:
            _save_joint_training_checkpoint(
                _joint_checkpoint_path(train_cfg.save_dir, step),
                units,
                optimizer,
                scaler,
                history,
                validation_metrics_history,
                train_tokens,
                restart_contract,
                step=step,
                device=device,
                save_dir=train_cfg.save_dir,
                ddp_integrity_history=ddp_state_digest_records,
            )

        if _should_run_periodic_step(step, train_cfg.log_every, 1):
            logger.info(
                (
                    "joint LM distillation step=%s/%s total=%.6f lm=%.6f "
                    "kl=%.6f topk_margin=%.6f hidden=%.6f "
                    "valid_objective=%.6f"
                ),
                step,
                train_cfg.max_steps,
                components["total"],
                components["lm"],
                components["kl"],
                components["teacher_topk_margin"],
                components["hidden"],
                history.valid_losses[-1],
            )

    if ddp_state_digest_enabled:
        _record_ddp_state_digest("post_timed_phase_before_best_checkpoint_restore")
    if restore_joint_best:
        _restore_best_replacement(config, units, device=device)
    if recovery_policy is not None:
        recovery_policy.write_report(train_cfg.save_dir, recovery_policy_history)
    if ddp_state_digest_enabled:
        _record_ddp_state_digest("post_best_checkpoint_restore")
    _save_token_stream_state(train_tokens, train_cfg.save_dir)
    if is_transformer_main_process():
        os.makedirs(train_cfg.save_dir, exist_ok=True)
        history_path = os.path.join(
            train_cfg.save_dir,
            "joint_validation_history.json",
        )
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema": "dendritic_joint_validation_history/v1",
                    "checkpoint_selection_metric": selection_metric,
                    "objective_selection_metric": (
                        "mean_objective"
                        if float(train_cfg.validation_tail_weight) == 0.0
                        else "mean_upper_tail_blend"
                    ),
                    "validation_global_samples": getattr(
                        train_cfg,
                        "validation_global_samples",
                        None,
                    ),
                    "validation_local_samples": int(train_cfg.valid_samples),
                    "validation_batches_per_rank": int(eval_batches),
                    "validation_allow_window_wrap": bool(
                        getattr(train_cfg, "validation_allow_window_wrap", True)
                    ),
                    "sequence_risk_weight": float(train_cfg.sequence_risk_weight),
                    "sequence_risk_fraction": float(train_cfg.sequence_risk_fraction),
                    "teacher_topk_margin_weight": float(
                        train_cfg.teacher_topk_margin_weight
                    ),
                    "teacher_topk_margin_k": int(train_cfg.teacher_topk_margin_k),
                    "teacher_topk_margin_epsilon": float(
                        train_cfg.teacher_topk_margin_epsilon
                    ),
                    "validation_tail_weight": float(train_cfg.validation_tail_weight),
                    "validation_tail_fraction": float(
                        train_cfg.validation_tail_fraction
                    ),
                    **(
                        {"joint_recovery_policy": recovery_policy.manifest}
                        if recovery_policy is not None
                        else {}
                    ),
                    "records": validation_metrics_history,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        if ddp_state_digest_records:
            digest_path = os.path.join(
                train_cfg.save_dir,
                "ddp_state_sync_audit.json",
            )
            with open(digest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema": "dendritic_transformer_ddp_state_sync/v1",
                        "status": "all_checked_ranks_exact",
                        "digest_every_steps": ddp_state_digest_every,
                        "qualification_steps": ddp_state_digest_qualification_steps,
                        "explicit_steps": list(ddp_state_digest_explicit_steps),
                        "resolved_optimizer_steps": list(ddp_state_digest_steps),
                        "local_finite_loss_component_checks": (
                            ddp_local_finite_loss_checks
                        ),
                        "distributed_finite_loss_component_checks": (
                            ddp_distributed_finite_loss_checks
                        ),
                        "records": ddp_state_digest_records,
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")
    result = TransformerReplacementTrainingResult(
        **history.as_result_kwargs(),
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
        execution_mode=(
            "trained" if len(history.train_losses) > 0 else "zero_shot_compile"
        ),
        steps_requested=int(train_cfg.max_steps),
        steps_executed=len(history.train_losses),
    )
    # Preserve a config-reconstructible train-time student before the per-layer
    # exporter optionally replaces dynamic sparse modules with fixed topology.
    _save_student_model(student, config)
    _save_result(
        result,
        units,
        save_replacements=train_cfg.save_replacements,
        topology_encoding=str(train_cfg.replacement_checkpoint_encoding),
        freeze_sparse_topology_on_export=bool(
            getattr(train_cfg, "freeze_sparse_topology_on_export", False)
        ),
        stochastic_topology_freeze_policy=str(
            getattr(train_cfg, "stochastic_topology_freeze_policy", "reject")
        ),
        stochastic_topology_freeze_seed=int(
            getattr(train_cfg, "stochastic_topology_freeze_seed", 0)
        ),
        ragged_topology_format=str(getattr(train_cfg, "ragged_topology_format", "csr")),
    )
    return result


def _run_transformer_replacement_training(
    config: Config,
    *,
    distributed_context,
) -> TransformerReplacementTrainingResult:
    """Run the configured transformer replacement distillation experiment."""
    train_cfg = config.training.transformer_replacement
    if not train_cfg.enabled:
        raise ValueError("training.transformer_replacement.enabled must be true")
    mode = str(train_cfg.mode).lower()
    validate_joint_recovery_policy_config(train_cfg)
    if mode not in {"layerwise_distillation", "joint_lm_distillation"}:
        raise ValueError(
            "mode must be 'layerwise_distillation' or 'joint_lm_distillation'"
        )
    tied_groups = list(
        getattr(
            config.model.transformer_replacement,
            "parameter_tied_replacement_groups",
            [],
        )
        or []
    )
    if tied_groups:
        if bool(getattr(train_cfg, "save_student_model", False)):
            raise NotImplementedError(
                "save_student_model is not alias-aware for shared replacement "
                "groups; use deduplicated replacement artifacts"
            )
        if validate_pruning_ladder_config(getattr(train_cfg, "pruning_ladder", None)):
            raise NotImplementedError(
                "pruning_ladder does not yet support parameter-tied replacement groups"
            )
        if mode == "layerwise_distillation" and (
            str(getattr(train_cfg, "distributed_mode", "none") or "none").lower()
            != "none"
            or bool(getattr(train_cfg, "data_parallel", False))
        ):
            raise NotImplementedError(
                "distributed layerwise training does not support shared "
                "replacement groups; use single-process training"
            )
    collapsed_spans = list(
        getattr(
            config.model.transformer_replacement,
            "collapsed_replacement_spans",
            [],
        )
        or []
    )
    if collapsed_spans:
        if (
            mode != "joint_lm_distillation"
            or str(train_cfg.teacher_source) != "hf_text"
        ):
            raise NotImplementedError(
                "collapsed replacement spans require joint_lm_distillation with "
                "teacher_source='hf_text' so training can supervise teacher "
                "span-exit hidden states"
            )
        if float(getattr(train_cfg, "hidden_loss_weight", 0.0)) <= 0:
            raise ValueError(
                "collapsed replacement spans require hidden_loss_weight > 0; "
                "span-exit supervision must not be silently disabled"
            )
        exits = [int(span[-1]) for span in collapsed_spans]
        configured_hidden = list(getattr(train_cfg, "hidden_loss_layers", []) or [])
        if configured_hidden and [int(layer) for layer in configured_hidden] != exits:
            raise ValueError(
                "collapsed replacement hidden_loss_layers must equal the ordered "
                f"span exits {exits}"
            )
    if (
        mode == "layerwise_distillation"
        and str(train_cfg.train_target).lower() != "replacement_only"
    ):
        raise ValueError(
            "layerwise_distillation supports only train_target='replacement_only'; "
            "use mode='joint_lm_distillation' for train_target='full_model'"
        )

    seed = int(train_cfg.seed if train_cfg.seed is not None else config.experiment.seed)
    seed += int(distributed_context.rank)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = (
        distributed_context.device
        if distributed_context.enabled
        else _resolve_device(train_cfg.device)
    )
    assert device is not None
    dtype = _resolve_dtype(train_cfg.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        logger.warning("Using float32 for CPU transformer replacement training")
        dtype = torch.float32

    source = str(train_cfg.teacher_source).lower()
    default_layers = [0] if source in {"synthetic_mlp", "hidden_cache"} else []
    replacement_layers = resolve_configured_replacement_layers(
        config,
        default=default_layers,
    )
    if not replacement_layers:
        raise ValueError("At least one transformer replacement layer is required")
    if mode == "joint_lm_distillation":
        if source != "hf_text":
            raise ValueError(
                "joint_lm_distillation currently requires teacher_source='hf_text'"
            )
        return _train_joint_hf_text_source(config, device=device, dtype=dtype)

    if source == "synthetic_mlp":
        units = _make_synthetic_units(config, device=device, dtype=dtype)
        train_datasets, valid_datasets = _build_synthetic_datasets(
            units,
            config,
            device=device,
            dtype=dtype,
        )
        return _train_tensor_dataset_source(
            units,
            train_datasets,
            valid_datasets,
            config,
            device=device,
            dtype=dtype,
        )
    if source == "hidden_cache":
        units, train_datasets, valid_datasets = _build_hidden_cache_units_and_datasets(
            config,
            device=device,
            dtype=dtype,
        )
        return _train_tensor_dataset_source(
            units,
            train_datasets,
            valid_datasets,
            config,
            device=device,
            dtype=dtype,
        )
    if source == "hf_text":
        return _train_hf_text_source(config, device=device, dtype=dtype)
    raise ValueError(
        "teacher_source must be 'synthetic_mlp', 'hidden_cache', or 'hf_text'"
    )


def run_transformer_replacement_training(
    config: Config,
) -> TransformerReplacementTrainingResult:
    """Run a transformer replacement experiment, optionally under torchrun DDP."""
    context = initialize_transformer_distributed(
        config.training.transformer_replacement
    )
    started = time.perf_counter()
    metrics_device = context.device or _resolve_device(
        config.training.transformer_replacement.device
    )
    use_cuda_metrics = metrics_device.type == "cuda"
    if use_cuda_metrics:
        torch.cuda.reset_peak_memory_stats()
    try:
        precision_receipt = _configure_transformer_execution_precision(
            config.experiment
        )
        if is_transformer_main_process():
            _write_json_atomic(
                Path(config.training.transformer_replacement.save_dir)
                / "precision_runtime.json",
                precision_receipt,
            )
        result = _run_transformer_replacement_training(
            config,
            distributed_context=context,
        )
        result.wall_time_seconds = time.perf_counter() - started
        if use_cuda_metrics:
            result.peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated()
            result.peak_cuda_reserved_bytes = torch.cuda.max_memory_reserved()
        if is_transformer_main_process():
            _write_metrics_json(result, output_dir=result.save_dir)
        return result
    finally:
        context.close()
